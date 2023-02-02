import torch
import torch.nn as nn
from typing import Any, List, Dict

from allennlp.nn.util import masked_mean
from tools import allennlp as util
from metrics.drop import DropEmAndF1
from metrics.tatqa import TaTQAEmAndF1

import model.counting_block as counting
import model.multiple_spans_wrapper as multiple_spans_wrapper

from model.bert import Bert
from model.postnet.reasoning import ReasoningBlock
from model.postnet import PostNet
from model.utils import (
    AnswerAbilityPredictor,
    AnswerOption,
    get_encoded_numbers_from_full_sequence,
)
from model.scale_predictor import ScaleBlock
from model.answering_modules.single_span.span_extractor import SpanExtractor
from model.answering_modules.single_span.span_extractor_2d import SpanExtractor2D
from model.addition_subtraction import AdditionSubtractionBlock
from model.average_predictor import AverageBlock
from model.change_ratio_predictor import ChangeRatioBlock
from model.division_predictor import DivisionBlock

from tools.config import Config

ANSWER_ABILITY_LOG_PROBS = "answer_ability_log_probs"


class AeNER(nn.Module):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.

    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """

    def __init__(
        self,
        config: Config,
        bert,
        answering_abilities: List[str] = None,
        unique_on_multispan: bool = True,
        multispan_use_bio_wordpiece_mask: bool = True,
        dont_add_substrings_to_ms: bool = True,
    ) -> None:
        super(AeNER, self).__init__()
        self.config = config
        self.bert = Bert(bert, config=config)

        if config.dataset in ["drop", "race", "squad"]:
            self.train_metrics = DropEmAndF1(config=config, is_train=True)
            self.eval_metrics = DropEmAndF1(config=config, is_train=False)
        elif config.dataset == "tatqa":
            self.train_metrics = TaTQAEmAndF1(config=config, is_train=True)
            self.eval_metrics = TaTQAEmAndF1(config=config, is_train=False)
        else:
            raise Exception("Unknown dataset name: {}".format(config.dataset))

        if answering_abilities is None:
            if config.dataset in ["drop", "race"]:
                self.answering_abilities = [
                    AnswerOption.SINGLE_SPAN,
                    AnswerOption.ADDITION_SUBTRACTION,
                    AnswerOption.COUNTING,
                    AnswerOption.MULTI_SPAN,
                ]
            elif config.dataset == "squad":
                self.answering_abilities = [AnswerOption.SINGLE_SPAN]
            elif config.dataset == "tatqa":
                self.answering_abilities = [
                    AnswerOption.SINGLE_SPAN,
                    AnswerOption.ADDITION_SUBTRACTION,
                    AnswerOption.AVERAGE,
                    AnswerOption.CHANGE_RATIO,
                    AnswerOption.DIVISION,
                    AnswerOption.MULTI_SPAN,
                ]
                if not config.counting_as_span:
                    self.answering_abilities.append(AnswerOption.COUNTING)
                else:
                    self.answering_abilities.append(AnswerOption.COUNTING_AS_MULTI_SPAN)
            else:
                raise Exception("Unknown dataset: {}".format(config.dataset))
        else:
            self.answering_abilities = answering_abilities

        if len(self.answering_abilities) > 1:
            self.answer_ability_predictor = AnswerAbilityPredictor(
                config.hidden_size, config.dropout, len(self.answering_abilities)
            )

        self.predict_scale = self.config.predict_scale
        if self.predict_scale:
            self.scale_predictor = ScaleBlock(config=config)

        if AnswerOption.SINGLE_SPAN in self.answering_abilities:
            if not config.span_2d:
                self.span_extractor = SpanExtractor(config=config)
            else:
                self.span_extractor = SpanExtractor2D(config=config)

        if (
            AnswerOption.MULTI_SPAN in self.answering_abilities
            or AnswerOption.COUNTING_AS_MULTI_SPAN in self.answering_abilities
        ):
            self.multispan_wrapper = multiple_spans_wrapper.MultipleSpanBlock(
                hidden_size=config.hidden_size,
                unique_on_multispan=unique_on_multispan,
                use_bio_wordpiece_mask=multispan_use_bio_wordpiece_mask,
                dont_add_substrings_to_ms=dont_add_substrings_to_ms,
            )

        if AnswerOption.ADDITION_SUBTRACTION in self.answering_abilities:
            self.addition_subtraction_block = AdditionSubtractionBlock(config)

        if AnswerOption.AVERAGE in self.answering_abilities:
            self.average_block = AverageBlock(config)

        if AnswerOption.CHANGE_RATIO in self.answering_abilities:
            self.change_ratio_block = ChangeRatioBlock(config)

        if AnswerOption.DIVISION in self.answering_abilities:
            self.division_block = DivisionBlock(config)

        if AnswerOption.COUNTING in self.answering_abilities:
            self.counting_block = counting.CountingBlock(config)

        self.use_reasoning = config.use_reasoning
        if self.use_reasoning:
            self.reasoner = ReasoningBlock(config=config)
        self.postnet = PostNet(config=config)

        # add bert proj
        in_hidden_size = config.hidden_size * (2 if config.use_heavy_postnet else 1)
        self.proj_number = nn.Linear(in_hidden_size, 1, bias=False)
        self.proj_sequence_h = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,  # type: ignore
        input_ids: torch.LongTensor,
        input_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        table_position_ids: torch.LongTensor,
        passage_mask: torch.LongTensor,
        question_mask: torch.LongTensor,
        span_mask: torch.LongTensor,
        passage_number_indices: torch.LongTensor,
        passage_number_order: torch.LongTensor,
        passage_numbers: torch.FloatTensor,
        question_number_order: torch.LongTensor,
        question_number_indices: torch.LongTensor,
        question_numbers: torch.FloatTensor,
        all_number_strs,
        answer_as_passage_spans: torch.LongTensor = None,
        answer_as_question_spans: torch.LongTensor = None,
        answer_as_add_sub_expressions: torch.LongTensor = None,
        answer_as_avg_expressions: torch.LongTensor = None,
        answer_as_change_ratio_expressions: torch.LongTensor = None,
        answer_as_div_expressions: torch.LongTensor = None,
        answer_as_counts: torch.LongTensor = None,
        answer_scales: torch.LongTensor = None,
        answer_as_text_to_disjoint_bios: torch.LongTensor = None,
        answer_as_list_of_bios: torch.LongTensor = None,
        span_bio_labels: torch.LongTensor = None,
        bio_wordpiece_mask: torch.LongTensor = None,
        is_bio_mask: torch.LongTensor = None,
        is_counts_multi_span: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        output_dict = {"metadata": metadata}

        batch_size = input_ids.size(0)
        sequence_output, sequence_output_list = self.bert(
            input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids
        )

        del input_ids, token_type_ids
        if self.config.use_heavy_postnet:
            sequence_output_or_list = sequence_output_list
        else:
            del sequence_output_list
            sequence_output_or_list = sequence_output

        reasoning_info_vec = None
        output_dict["num_emb_loss"] = torch.tensor(0.0, device=self.config.device)
        if self.use_reasoning:
            reasoning_info_vec, number_value_embedding_loss = self.reasoner(
                sequence_output_or_list=sequence_output_or_list,
                batch_size=batch_size,
                passage_number_indices=passage_number_indices,
                question_number_indices=question_number_indices,
                passage_number_order=passage_number_order,
                question_number_order=question_number_order,
                passage_numbers=passage_numbers,
                question_numbers=question_numbers,
                all_number_strs=all_number_strs,
            )
            output_dict["num_emb_loss"] = number_value_embedding_loss

        # Updates sequence_output_list
        # sequence_output_list.shape: (batch_size, len, hidden_size)
        # and does not change
        sequence_output_or_list = self.postnet(
            sequence_output_or_list,
            reasoning_info_vec,
            mask=input_mask,
            table_position_ids=table_position_ids,
            passage_number_indices=passage_number_indices,
        )

        if self.config.use_heavy_postnet:
            # sequence_output = sequence_output
            sequence_output_list = sequence_output_or_list
        else:
            sequence_output = sequence_output_or_list
            sequence_output_list = None

        # passage hidden and question hidden
        encoded_passage_for_text = (
            sequence_output_list[2]
            if (self.config.use_heavy_postnet)
            else sequence_output
        )
        sequence_h_weight = self.proj_sequence_h(encoded_passage_for_text).squeeze(-1)
        passage_h_weight = util.masked_softmax(sequence_h_weight, passage_mask)
        passage_h = util.weighted_sum(encoded_passage_for_text, passage_h_weight)
        question_h_weight = util.masked_softmax(sequence_h_weight, question_mask)
        question_h = util.weighted_sum(encoded_passage_for_text, question_h_weight)

        # numbers in passage hidden
        encoded_passage_for_numbers = (
            torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=-1)
            if self.config.use_heavy_postnet
            else sequence_output
        )
        _, encoded_passage_numbers = get_encoded_numbers_from_full_sequence(
            passage_number_indices, encoded_passage_for_numbers
        )
        # Adding dummy 100-value number (for percentage questions).
        passage_number_mask = (passage_number_indices > -2).long()
        passage_number_weight = self.proj_number(encoded_passage_numbers).squeeze(-1)
        passage_number_weight = util.masked_softmax(
            passage_number_weight, passage_number_mask
        )
        passage_number_vector = util.weighted_sum(
            encoded_passage_numbers, passage_number_weight
        )

        predictions = {}
        output_dict["predictions"] = predictions
        if len(self.answering_abilities) > 1:
            predictions[ANSWER_ABILITY_LOG_PROBS] = self.answer_ability_predictor(
                sequence_output, passage_h, question_h
            )
        if AnswerOption.COUNTING in self.answering_abilities:
            predictions[AnswerOption.COUNTING] = self.counting_block(
                passage_number_vector, passage_h, question_h, sequence_output
            )
        if AnswerOption.SINGLE_SPAN in self.answering_abilities:
            predictions[AnswerOption.SINGLE_SPAN] = self.span_extractor(
                sequence_output_or_list=sequence_output_or_list,
                question_mask=question_mask,
                passage_mask=passage_mask,
                span_mask=span_mask,
            )
        if (
            AnswerOption.MULTI_SPAN in self.answering_abilities
            or AnswerOption.COUNTING_AS_MULTI_SPAN in self.answering_abilities
        ):
            predictions[AnswerOption.MULTI_SPAN] = self.multispan_wrapper(
                sequence_output, input_mask, bio_wordpiece_mask
            )
        if AnswerOption.ADDITION_SUBTRACTION in self.answering_abilities:
            predictions[
                AnswerOption.ADDITION_SUBTRACTION
            ] = self.addition_subtraction_block(
                encoded_passage_numbers,
                question_h,
                passage_h,
                sequence_output,
                passage_number_mask,
            )
        if AnswerOption.AVERAGE in self.answering_abilities:
            predictions[AnswerOption.AVERAGE] = self.average_block(
                encoded_passage_numbers,
                question_h,
                passage_h,
                sequence_output,
                passage_number_mask,
            )
        if AnswerOption.CHANGE_RATIO in self.answering_abilities:
            predictions[AnswerOption.CHANGE_RATIO] = self.change_ratio_block(
                encoded_passage_numbers,
                question_h,
                passage_h,
                sequence_output,
                passage_number_mask,
            )
        if AnswerOption.DIVISION in self.answering_abilities:
            predictions[AnswerOption.DIVISION] = self.division_block(
                encoded_passage_numbers,
                question_h,
                passage_h,
                sequence_output,
                passage_number_mask,
            )

        with torch.no_grad():
            scale_mask = None
            if self.config.predict_scale:
                scale_mask = (
                    torch.LongTensor(input_mask.size()).to(self.config.device).fill_(0)
                )
                # We use CLS token.
                scale_mask[:, 0] = 1

            output_dict["answer_jsons"] = []
            if len(self.answering_abilities) >= 1:
                best_answer_ability = (
                    torch.argmax(predictions[ANSWER_ABILITY_LOG_PROBS], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            i = 0
            while i < batch_size:
                answer_json: dict[str, Any] = {}

                if len(self.answering_abilities) > 1:
                    answer_index = best_answer_ability[i]
                    predicted_ability = self.answering_abilities[answer_index]
                else:
                    predicted_ability = self.answering_abilities[0]
                answer_json["predicted_operation"] = predicted_ability

                if predicted_ability == AnswerOption.SINGLE_SPAN:
                    self.span_extractor.update_answer(
                        answer_json,
                        predictions[predicted_ability],
                        metadata,
                        i,
                        scale_mask=scale_mask,
                    )
                elif predicted_ability == AnswerOption.ADDITION_SUBTRACTION:
                    self.addition_subtraction_block.update_answer(
                        answer_json,
                        predictions[predicted_ability].best_signs[i],
                        predictions[predicted_ability].log_probs[i],
                        metadata[i],
                        scale_mask=scale_mask[i] if scale_mask is not None else None,
                    )
                elif predicted_ability == AnswerOption.AVERAGE:
                    self.average_block.update_answer(
                        answer_json,
                        predictions[predicted_ability].best_signs[i],
                        predictions[predicted_ability].log_probs[i],
                        metadata[i],
                        scale_mask=scale_mask[i] if scale_mask is not None else None,
                    )
                elif predicted_ability == AnswerOption.CHANGE_RATIO:
                    self.change_ratio_block.update_answer(
                        answer_json,
                        predictions[predicted_ability].best_signs[i],
                        predictions[predicted_ability].log_probs[i],
                        metadata[i],
                        scale_mask=scale_mask[i] if scale_mask is not None else None,
                    )
                elif predicted_ability == AnswerOption.DIVISION:
                    self.division_block.update_answer(
                        answer_json,
                        predictions[predicted_ability].best_signs[i],
                        predictions[predicted_ability].log_probs[i],
                        metadata[i],
                        scale_mask=scale_mask[i] if scale_mask is not None else None,
                    )
                elif predicted_ability == AnswerOption.COUNTING:
                    counting.update_answer(
                        answer_json, predictions[predicted_ability].best_number[i]
                    )
                elif predicted_ability == AnswerOption.COUNTING_AS_MULTI_SPAN:
                    multiple_spans_wrapper.update_answer(
                        self.multispan_wrapper,
                        predictions[AnswerOption.MULTI_SPAN],
                        i,
                        metadata,
                        answer_json,
                        bio_wordpiece_mask,
                        is_counts_multi_span=True,
                        scale_mask=scale_mask,
                    )
                elif predicted_ability == AnswerOption.MULTI_SPAN:
                    multiple_spans_wrapper.update_answer(
                        self.multispan_wrapper,
                        predictions[predicted_ability],
                        i,
                        metadata,
                        answer_json,
                        bio_wordpiece_mask,
                        is_counts_multi_span=False,
                        scale_mask=scale_mask,
                    )
                    if len(answer_json["predicted_answer"]) == 0:
                        # Second-best answer option.
                        best_answer_ability[i] = torch.topk(
                            predictions[ANSWER_ABILITY_LOG_PROBS], k=2, dim=1
                        )[1][i][1]
                        continue
                else:
                    raise ValueError(f"Unsupported answer ability: {predicted_ability}")
                output_dict["answer_jsons"].append(answer_json)
                i += 1

        if self.predict_scale:
            answer_h = masked_mean(
                encoded_passage_for_text, scale_mask.bool().unsqueeze(-1), dim=1
            )
            predictions["scale"] = self.scale_predictor(
                passage_number_vector, passage_h, question_h, answer_h, sequence_output
            )

        with torch.no_grad():
            for i in range(batch_size):
                answer_json = output_dict["answer_jsons"][i]
                if self.predict_scale:
                    self.scale_predictor.update_answer(
                        answer_json, predictions["scale"].best_scale[i]
                    )
                answer_annotations = metadata[i].get("answer_annotations", [])
                metrics = self.train_metrics if self.training else self.eval_metrics
                metrics(
                    prediction=answer_json,
                    ground_truth=answer_annotations,
                    question_id=metadata[i]["question_id"],
                    question_text=metadata[i]["original_question"],
                    passage_text=metadata[i]["original_passage"],
                    metas_logits=output_dict,
                    idx_in_batch=i,
                )

        self.calculate_loss(
            output_dict=output_dict,
            predictions=predictions,
            passage_number_mask=passage_number_mask,
            answer_as_passage_spans=answer_as_passage_spans,
            answer_as_question_spans=answer_as_question_spans,
            answer_as_add_sub_expressions=answer_as_add_sub_expressions,
            answer_as_avg_expressions=answer_as_avg_expressions,
            answer_as_change_ratio_expressions=answer_as_change_ratio_expressions,
            answer_as_div_expressions=answer_as_div_expressions,
            answer_as_counts=answer_as_counts,
            answer_as_list_of_bios=answer_as_list_of_bios,
            answer_as_text_to_disjoint_bios=answer_as_text_to_disjoint_bios,
            span_bio_labels=span_bio_labels,
            is_bio_mask=is_bio_mask,
            is_counts_multi_span=is_counts_multi_span,
            answer_scales=answer_scales,
        )

        return output_dict

    def get_train_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.train_metrics.get_metric(reset)

    def get_eval_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.eval_metrics.get_metric(reset)

    def calculate_loss(
        self,
        output_dict,
        predictions,
        passage_number_mask,
        answer_as_passage_spans,
        answer_as_question_spans,
        answer_as_add_sub_expressions,
        answer_as_avg_expressions,
        answer_as_change_ratio_expressions,
        answer_as_div_expressions,
        answer_as_counts,
        answer_as_list_of_bios,
        answer_as_text_to_disjoint_bios,
        span_bio_labels,
        is_bio_mask,
        is_counts_multi_span,
        answer_scales,
    ):
        # If answer is given, compute the loss.
        if (
            answer_as_passage_spans is not None
            or answer_as_question_spans is not None
            or answer_as_add_sub_expressions is not None
            or answer_as_avg_expressions is not None
            or answer_as_change_ratio_expressions is not None
            or answer_as_div_expressions is not None
            or answer_as_counts is not None
            or answer_as_list_of_bios is not None
            or answer_as_text_to_disjoint_bios is not None
        ):
            log_marginal_likelihood_list = []
            for answering_ability in self.answering_abilities:
                if answering_ability == AnswerOption.SINGLE_SPAN:
                    log_probs = torch.stack(
                        [
                            self.span_extractor.calc_log_marginal_likelihood(
                                answer_as_question_spans,
                                predictions[AnswerOption.SINGLE_SPAN],
                            ),
                            self.span_extractor.calc_log_marginal_likelihood(
                                answer_as_passage_spans,
                                predictions[AnswerOption.SINGLE_SPAN],
                            ),
                        ],
                        dim=-1,
                    )
                    log_marginal_likelihood_list.append(util.logsumexp(log_probs))
                elif answering_ability == AnswerOption.ADDITION_SUBTRACTION:
                    log_marginal_likelihood_list.append(
                        AdditionSubtractionBlock.calc_log_marginal_likelihood(
                            answer_as_add_sub_expressions,
                            predictions[AnswerOption.ADDITION_SUBTRACTION],
                            passage_number_mask,
                        )
                    )
                elif answering_ability == AnswerOption.AVERAGE:
                    log_marginal_likelihood_list.append(
                        AverageBlock.calc_log_marginal_likelihood(
                            answer_as_avg_expressions,
                            predictions[AnswerOption.AVERAGE],
                            passage_number_mask,
                        )
                    )
                elif answering_ability == AnswerOption.CHANGE_RATIO:
                    log_marginal_likelihood_list.append(
                        ChangeRatioBlock.calc_log_marginal_likelihood(
                            answer_as_change_ratio_expressions,
                            predictions[AnswerOption.CHANGE_RATIO],
                            passage_number_mask,
                        )
                    )
                elif answering_ability == AnswerOption.DIVISION:
                    log_marginal_likelihood_list.append(
                        DivisionBlock.calc_log_marginal_likelihood(
                            answer_as_div_expressions,
                            predictions[AnswerOption.DIVISION],
                            passage_number_mask,
                        )
                    )
                elif answering_ability == AnswerOption.COUNTING:
                    log_marginal_likelihood_list.append(
                        counting.calc_log_marginal_likelihood(
                            answer_as_counts, predictions[AnswerOption.COUNTING]
                        )
                    )
                elif answering_ability == AnswerOption.COUNTING_AS_MULTI_SPAN:
                    log_marginal_likelihood_list.append(
                        self.multispan_wrapper.calc_log_marginal_likelihood(
                            prediction=predictions[AnswerOption.MULTI_SPAN],
                            answer_as_text_to_disjoint_bios=answer_as_text_to_disjoint_bios,
                            answer_as_list_of_bios=answer_as_list_of_bios,
                            span_bio_labels=span_bio_labels,
                            is_bio_mask=is_bio_mask,
                        )
                        + is_counts_multi_span.log()
                    )
                elif answering_ability == AnswerOption.MULTI_SPAN:
                    log_marginal_likelihood_list.append(
                        self.multispan_wrapper.calc_log_marginal_likelihood(
                            prediction=predictions[AnswerOption.MULTI_SPAN],
                            answer_as_text_to_disjoint_bios=answer_as_text_to_disjoint_bios,
                            answer_as_list_of_bios=answer_as_list_of_bios,
                            span_bio_labels=span_bio_labels,
                            is_bio_mask=is_bio_mask,
                        )
                        + (1 - is_counts_multi_span).log()
                    )
                else:
                    raise ValueError(
                        f"Unsupported answering ability: {answering_ability}"
                    )

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities.
                all_log_marginal_likelihoods = (
                    torch.stack(log_marginal_likelihood_list, dim=-1)
                    + predictions[ANSWER_ABILITY_LOG_PROBS]
                )
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
            if self.predict_scale:
                marginal_log_likelihood = (
                    marginal_log_likelihood
                    + self.scale_predictor.calc_log_marginal_likelihood(
                        answer_scales, predictions["scale"]
                    )
                )
            output_dict["loss"] = -marginal_log_likelihood.mean()
