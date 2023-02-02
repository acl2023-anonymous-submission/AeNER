# add multiple span prediction from https://github.com/eladsegal/project-NLP-AML

from collections import OrderedDict
import torch.nn as nn
from model.multispan_heads import FlexibleLoss
from model.multispan_heads import remove_substring_from_prediction


class MultipleSpanResult:
    def __init__(self, multispan_mask, logits, log_probs) -> None:
        self.logits = logits
        self.multispan_mask = multispan_mask
        self.log_probs = log_probs


class MultipleSpanBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        unique_on_multispan,
        use_bio_wordpiece_mask,
        dont_add_substrings_to_ms,
    ) -> None:
        super(MultipleSpanBlock, self).__init__()

        self.unique_on_multispan = unique_on_multispan
        self.use_bio_wordpiece_mask = use_bio_wordpiece_mask
        self.dont_add_substrings_to_ms = dont_add_substrings_to_ms
        self.head = FlexibleLoss(hidden_size)

    def forward(self, sequence_output, input_mask, bio_wordpiece_mask):
        # add multiple span prediction
        if bio_wordpiece_mask is None or not self.use_bio_wordpiece_mask:
            multispan_mask = input_mask
        else:
            multispan_mask = input_mask * bio_wordpiece_mask

        log_probs, logits = self.head.module(
            bert_out=sequence_output, seq_mask=multispan_mask
        )

        return MultipleSpanResult(multispan_mask, logits, log_probs)

    def calc_log_marginal_likelihood(
        self,
        prediction: MultipleSpanResult,
        answer_as_text_to_disjoint_bios,
        answer_as_list_of_bios,
        span_bio_labels,
        is_bio_mask,
    ):
        log_marginal_likelihood = self.head.log_likelihood(
            answer_as_text_to_disjoint_bios=answer_as_text_to_disjoint_bios,
            answer_as_list_of_bios=answer_as_list_of_bios,
            span_bio_labels=span_bio_labels,
            log_probs=prediction.log_probs,
            seq_mask=prediction.multispan_mask,
            is_bio_mask=is_bio_mask,
        )
        return log_marginal_likelihood


def update_answer(
    self: MultipleSpanBlock,
    prediction: MultipleSpanResult,
    i,
    metadata,
    answer_json,
    bio_wordpiece_mask,
    is_counts_multi_span,
    scale_mask,
):
    passage_str = metadata[i]["original_passage"]
    question_str = metadata[i]["original_question"]
    qp_tokens = metadata[i]["question_passage_tokens"]

    (
        answer_json["span_texts"],
        answer_json["predicted_spans"],
        answer_json["predicted_token_indices_bounds"],
    ) = self.head.prediction(
        logits=prediction.logits[i],
        qp_tokens=qp_tokens,
        p_text=passage_str,
        q_text=question_str,
        seq_mask=prediction.multispan_mask[i],
        wordpiece_mask=bio_wordpiece_mask[i],
    )

    if scale_mask is not None:
        for s, e in answer_json["predicted_token_indices_bounds"]:
            scale_mask[i][s:e] = 1

    if self.unique_on_multispan:
        answer_json["span_texts"] = list(
            OrderedDict.fromkeys(answer_json["span_texts"])
        )

        if self.dont_add_substrings_to_ms:
            answer_json["span_texts"] = remove_substring_from_prediction(
                answer_json["span_texts"]
            )

    if is_counts_multi_span:
        answer_json["predicted_answer"] = str(len(answer_json["span_texts"]))
    else:
        answer_json["predicted_answer"] = answer_json["span_texts"]
