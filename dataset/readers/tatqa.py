import json
import itertools
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

from metrics.tatqa_utils import normalize_answer

from dataset.readers import Reader

from dataset.utils import (
    create_bio_labels,
    get_number_order,
    find_valid_add_sub_expressions,
    find_valid_change_ratio_expressins,
    find_valid_div_expressions,
    restore_expressions_from_filtered,
)
from dataset.tokenization import (
    prepare_tokens,
    roberta_extract_numbers,
    get_number_from_text,
)
from dataset.utils import ANSWER_SCALE, ID_TO_ANSWER_SCALE, PERCENT_SCALE
from model.utils import AnswerOption
from tools.config import Config
from tools.utils import convert_number_to_str


class TatQAReader(Reader):
    def __init__(
        self, tokenizer, config: Config, skip_when_all_empty: List[str] = None
    ) -> None:
        super(TatQAReader, self).__init__(
            tokenizer,
            config,
            normalize_answer_function=normalize_answer,
            skip_when_all_empty=skip_when_all_empty,
        )
        for item in self.skip_when_all_empty:
            assert item in [
                "passage_span",
                "question_span",
                "addition_subtraction",
                "average",
                "change_ratio",
                "division",
                "counting",
                "multi_span",
            ], f"Unsupported skip type: {item}"

    def _read(self, file_path: str):
        print("Reading file at", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances, skip_count, cuts = [], 0, 0
        max_y_table, max_x_table = 0, 0
        for passage_info in tqdm(dataset, desc="Reading the dataset"):

            passage_ids = []
            table = passage_info["table"]["table"]
            max_y_table = max(max_y_table, len(table))
            max_x_table = max(max_x_table, len(table[0]))
            passage_texts = []
            unique_passage_text = set()
            for passage in passage_info["paragraphs"]:
                if passage["text"] in unique_passage_text:
                    print("Duplicate found")
                    continue
                unique_passage_text.add(passage["text"])
                passage_ids.append(passage["uid"])
                passage_texts.append(passage["text"])

            for question_answer in passage_info["questions"]:
                question_id = question_answer["uid"]
                question_text = question_answer["question"]

                answer_annotations = []
                if "answer" in question_answer:
                    derivation = None
                    if "derivation" in question_answer:
                        derivation = question_answer["derivation"]
                    answer_annotations.append(
                        {
                            "answer_type": question_answer["answer_type"],
                            "answer": question_answer["answer"],
                            "scale": question_answer["scale"],
                            "derivation": derivation,
                        }
                    )

                instance, is_cut = self.text_to_instance(
                    question_text,
                    table,
                    passage_texts,
                    question_id,
                    passage_ids,
                    answer_annotations,
                )
                cuts += is_cut
                if instance is not None:
                    instances.append(instance)
                else:
                    skip_count += 1

        print(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        print(f"Cut {cuts} samples.")
        print("Max table size:", (max_y_table, max_x_table))
        return instances

    def text_to_instance(
        self,
        question_text: str,
        table: List[List[str]],
        passage_texts: str,
        question_id: str,
        passage_ids: str,
        answer_annotations,
    ):
        (
            passage_text,
            passage_tokens,
            passage_offsets,
            passage_numbers,
            passage_number_indices,
            passage_text_segments,
            question_text,
            question_tokens,
            question_offsets,
            question_numbers,
            question_number_indices,
            question_text_segments,
            qp_tokens,
            qp_wordpiece_mask,
            qp_table_position,
            qp_text_segments,
            is_cut,
        ) = prepare_tokens(
            question_text, passage_texts, table, self.tokenizer, config=self.config
        )

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            answer_type, answer_texts = self.extract_answer_info_from_annotation(
                answer_annotations[0]
            )
            derivation: str = answer_annotations[0]["derivation"]

        # Scale for Tat-QA.
        valid_scales = []
        if answer_annotations:
            valid_scales = [ANSWER_SCALE[answer_annotations[0]["scale"]]]

        target_numbers = []
        if answer_type in ["count", "arithmetic"]:
            # `answer_texts` is a list of valid answers.
            for answer_text in answer_texts:
                number = get_number_from_text(answer_text)
                if number is not None:
                    target_numbers.append(number)

        specific_answer_type_is_multispan = False
        valid_passage_spans = []
        valid_question_spans = []
        if answer_type in ["span", "multi-span", "count"]:
            if answer_type == "count":
                answer_texts = derivation.split("##")
            answer_texts = list(OrderedDict.fromkeys(answer_texts))
            if answer_type in ["multi-span", "count"]:
                specific_answer_type_is_multispan = True
            valid_passage_spans = (
                self.find_valid_spans(
                    passage_text, passage_offsets, answer_texts, passage_text_segments
                )
                if answer_texts
                else []
            )
            if len(valid_passage_spans) > 0:
                valid_question_spans = []
            else:
                valid_question_spans = (
                    self.find_valid_spans(
                        question_text,
                        question_offsets,
                        answer_texts,
                        question_text_segments,
                    )
                    if answer_texts
                    else []
                )

            if len(valid_question_spans):
                valid_question_spans = []
                print("Question span:", answer_texts)

        if self.config.no_single_span:
            specific_answer_type_is_multispan = True

        all_number = passage_numbers + question_numbers
        all_number_order = get_number_order(all_number)

        if all_number_order is None:
            passage_number_order = []
            question_number_order = []
        else:
            passage_number_order = all_number_order[: len(passage_numbers)]
            question_number_order = all_number_order[len(passage_numbers) :]

        passage_number_order = np.array(passage_number_order)
        question_number_order = np.array(question_number_order)

        valid_signs_for_add_sub_expressions: List[List[int]] = []
        valid_signs_for_avg_expressions: List[List[int]] = []
        valid_signs_for_change_ratio_expressions: List[List[int]] = []
        valid_signs_for_div_expressions: List[List[int]] = []
        if answer_type in ["arithmetic"]:
            target_number_strs = [
                convert_number_to_str(num, self.config) for num in target_numbers
            ]

            numbers_whitelist = set()
            if derivation is not None:
                derivation_numbers = roberta_extract_numbers(
                    config=self.config, text=derivation, tokenizer=self.tokenizer
                )
                for number in derivation_numbers:
                    numbers_whitelist.add(convert_number_to_str(number, self.config))

            passage_numbers_filtered = []
            filter_correspondence = []
            for i, number in enumerate(passage_numbers):
                if convert_number_to_str(number, self.config) in numbers_whitelist:
                    passage_numbers_filtered.append(number)
                    filter_correspondence.append(i)

            sum_expression_candidates = find_valid_add_sub_expressions(
                passage_numbers_filtered, target_number_strs, config=self.config
            )
            valid_signs_for_add_sub_expressions = restore_expressions_from_filtered(
                sum_expression_candidates, filter_correspondence, len(passage_numbers)
            )
            if "/" in derivation:
                valid_signs_for_add_sub_expressions = []

            avg_expression_candidates = find_valid_add_sub_expressions(
                passage_numbers_filtered,
                target_number_strs,
                config=self.config,
                avg=True,
                max_number_of_numbers_to_consider=5,
            )
            valid_signs_for_avg_expressions = restore_expressions_from_filtered(
                avg_expression_candidates, filter_correspondence, len(passage_numbers)
            )
            if "/" not in derivation:
                valid_signs_for_avg_expressions = []

            change_ratio_expression_candidates = find_valid_change_ratio_expressins(
                passage_numbers_filtered, target_number_strs, config=self.config
            )
            valid_signs_for_change_ratio_expressions = (
                restore_expressions_from_filtered(
                    change_ratio_expression_candidates,
                    filter_correspondence,
                    len(passage_numbers),
                )
            )
            if (
                ID_TO_ANSWER_SCALE[valid_scales[0]] != PERCENT_SCALE
                or "/" not in derivation
            ):
                valid_signs_for_change_ratio_expressions = []

            div_expression_candidates = find_valid_div_expressions(
                passage_numbers_filtered,
                target_number_strs,
                config=self.config,
                scale=ID_TO_ANSWER_SCALE[valid_scales[0]],
                max_number_of_numbers_to_consider=3,
            )
            valid_signs_for_div_expressions = restore_expressions_from_filtered(
                div_expression_candidates, filter_correspondence, len(passage_numbers)
            )
            if (
                ID_TO_ANSWER_SCALE[valid_scales[0]] not in ["", PERCENT_SCALE]
                or "average number" in question_text
                or "/" not in derivation
            ):
                valid_signs_for_div_expressions = []

            if (
                (
                    "percentage change" in question_text
                    or "percentage increase" in question_text
                    or "percentage decrease" in question_text
                )
                and not (
                    "absolute percentage change" in question_text
                    or "absolute percentage increase" in question_text
                    or "absolute percentage decrease" in question_text
                )
                and valid_signs_for_change_ratio_expressions
            ):
                valid_signs_for_add_sub_expressions = []
                valid_signs_for_avg_expressions = []
                valid_signs_for_div_expressions = []

            if "average" in question_text and valid_signs_for_avg_expressions:
                valid_signs_for_add_sub_expressions = []
                valid_signs_for_change_ratio_expressions = []
                valid_signs_for_div_expressions = []

            if "/" in derivation and valid_signs_for_div_expressions:
                valid_signs_for_add_sub_expressions = []
                valid_signs_for_change_ratio_expressions = []
                valid_signs_for_avg_expressions = []

            if valid_signs_for_add_sub_expressions:
                valid_signs_for_avg_expressions = []
                valid_signs_for_change_ratio_expressions = []
                valid_signs_for_div_expressions = []

            if (
                valid_signs_for_div_expressions
                and ID_TO_ANSWER_SCALE[valid_scales[0]] == PERCENT_SCALE
            ):
                valid_scales = [ANSWER_SCALE[""]]

        valid_counts: List[int] = []
        if answer_type in ["count"]:
            # Currently we only support count number 0 ~ 9
            numbers_for_count = list(range(10))
            valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

        # add multi_span answer extraction
        no_answer_bios = [0] * len(qp_tokens)
        if specific_answer_type_is_multispan and (
            len(valid_passage_spans) > 0 or len(valid_question_spans) > 0
        ):
            spans_dict = {}
            text_to_disjoint_bios = []
            flexibility_count = 1
            for answer_text in answer_texts:
                qspans = self.find_valid_spans(
                    question_text,
                    question_offsets,
                    [answer_text],
                    question_text_segments,
                )
                pspans = self.find_valid_spans(
                    passage_text, passage_offsets, [answer_text], passage_text_segments
                )
                pspans = [
                    (el[0] + len(question_tokens) + 2, el[1] + len(question_tokens) + 2)
                    for el in pspans
                ]
                spans = qspans + pspans
                if len(spans) == 0:
                    # possible if the passage was clipped, but not for all of the answers
                    continue
                spans_dict[answer_text] = spans

                disjoint_bios = []
                for span_ind, span in enumerate(spans):
                    bios = create_bio_labels([span], len(qp_tokens))
                    disjoint_bios.append(bios)

                text_to_disjoint_bios.append(disjoint_bios)
                flexibility_count *= (2 ** len(spans)) - 1

            answer_as_text_to_disjoint_bios = text_to_disjoint_bios

            if flexibility_count < self.flexibility_threshold:
                # Generate all non-empty span combinations per each text.
                spans_combinations_dict = {}
                for key, spans in spans_dict.items():
                    spans_combinations_dict[key] = all_combinations = []
                    for i in range(1, len(spans) + 1):
                        all_combinations += list(itertools.combinations(spans, i))

                # Calculate product between all the combinations per each text.
                packed_gold_spans_list = itertools.product(
                    *list(spans_combinations_dict.values())
                )
                bios_list = []
                for packed_gold_spans in packed_gold_spans_list:
                    gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                    bios = create_bio_labels(gold_spans, len(qp_tokens))
                    bios_list.append(bios)

                answer_as_list_of_bios = bios_list
                answer_as_text_to_disjoint_bios = [[no_answer_bios]]
            else:
                answer_as_list_of_bios = [no_answer_bios]

            # Used for both "require-all" BIO loss and flexible loss.
            bio_labels = create_bio_labels(
                valid_question_spans + valid_passage_spans, len(qp_tokens)
            )
            span_bio_labels = bio_labels

            is_bio_mask = 1

            multi_span = [
                is_bio_mask,
                answer_as_text_to_disjoint_bios,
                answer_as_list_of_bios,
                span_bio_labels,
            ]
        else:
            multi_span = []

        if specific_answer_type_is_multispan:
            valid_passage_spans = []
            valid_question_spans = []

        if answer_type == "count":
            if multi_span == []:
                valid_counts = []
                print("Bad count: no multispan")
            elif valid_counts == []:
                multi_span = []
                print("Bad count: {}".format(target_numbers))

        type_to_answer_map = {
            "passage_span": valid_passage_spans,
            "question_span": valid_question_spans,
            "addition_subtraction": valid_signs_for_add_sub_expressions,
            "average": valid_signs_for_avg_expressions,
            "change_ratio": valid_signs_for_change_ratio_expressions,
            "division": valid_signs_for_div_expressions,
            "counting": valid_counts,
            "multi_span": multi_span,
        }

        if self.skip_when_all_empty and not any(
            type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty
        ):
            return None, is_cut

        if answer_annotations:
            answer_annotations[0]["scale_processed"] = ID_TO_ANSWER_SCALE[
                valid_scales[0]
            ]
            if valid_passage_spans:
                answer_annotations[0][
                    "answer_type_processed"
                ] = AnswerOption.PASSAGE_SPAN
            elif valid_question_spans:
                answer_annotations[0][
                    "answer_type_processed"
                ] = AnswerOption.QUESTION_SPAN
            elif valid_signs_for_add_sub_expressions:
                answer_annotations[0][
                    "answer_type_processed"
                ] = AnswerOption.ADDITION_SUBTRACTION
            elif valid_signs_for_avg_expressions:
                answer_annotations[0]["answer_type_processed"] = AnswerOption.AVERAGE
            elif valid_signs_for_change_ratio_expressions:
                answer_annotations[0][
                    "answer_type_processed"
                ] = AnswerOption.CHANGE_RATIO
            elif valid_signs_for_div_expressions:
                answer_annotations[0]["answer_type_processed"] = AnswerOption.DIVISION
            elif valid_counts:
                answer_annotations[0]["answer_type_processed"] = AnswerOption.COUNTING
            elif multi_span and answer_type != "count":
                answer_annotations[0]["answer_type_processed"] = AnswerOption.MULTI_SPAN
            elif multi_span and answer_type == "count":
                answer_annotations[0][
                    "answer_type_processed"
                ] = AnswerOption.COUNTING_AS_MULTI_SPAN
            else:
                del answer_annotations[0]["scale_processed"]

        metadata = {
            "question_tokens": [token for token in question_tokens],
            "passage_tokens": [token for token in passage_tokens],
            "question_passage_tokens": qp_tokens,
            "passage_number_indices": passage_number_indices,
            "question_number_indices": question_number_indices,
            "passage_number_order": passage_number_order,
            "question_number_order": question_number_order,
            "wordpiece_mask": qp_wordpiece_mask,
            "table_position": qp_table_position,
            "text_segments": qp_text_segments,
            "original_passage": passage_text,
            "passage_token_offsets": passage_offsets,
            "original_question": question_text,
            "question_token_offsets": question_offsets,
            "passage_numbers": passage_numbers,
            "question_numbers": question_numbers,
            "passage_ids": passage_ids,
            "question_id": question_id,
            "answer_annotations": answer_annotations,
            "answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
            "answer_passage_spans": valid_passage_spans,
            "answer_question_spans": valid_question_spans,
            "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions,
            "signs_for_avg_expressions": valid_signs_for_avg_expressions,
            "signs_for_change_ratio_expressions": valid_signs_for_change_ratio_expressions,
            "signs_for_div_expressions": valid_signs_for_div_expressions,
            "counts": valid_counts,
            "multi_span": multi_span,
            "is_counts_multi_span": answer_type == "count",
            "scales": valid_scales,
        }

        return metadata, is_cut

    @staticmethod
    def extract_answer_info_from_annotation(
        answer_annotation: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        answer_type = answer_annotation["answer_type"]
        if answer_type == "span" or answer_type == "multi-span":
            answer_texts = []
            for answer_text in answer_annotation["answer"]:
                if answer_text != "":
                    answer_texts.append(answer_text)
                else:
                    print('Detected bad answer text: "{}"'.format(answer_text))
        elif answer_type == "count":
            answer_texts = [answer_annotation["answer"]]
        elif answer_type == "arithmetic":
            answer_texts = [str(answer_annotation["answer"])]
        else:
            raise "Unknwon answer type"

        return answer_type, answer_texts

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices
