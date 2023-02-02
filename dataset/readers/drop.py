import json
import itertools
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

from metrics.drop_utils import _normalize_answer

from dataset.readers import Reader

from dataset.utils import (
    get_number_order,
    create_bio_labels,
    find_valid_add_sub_expressions,
)

from dataset.tokenization import prepare_tokens, get_number_from_text

from model.utils import AnswerOption

from tools.config import Config
from tools.utils import convert_number_to_str


class DropReader(Reader):
    def __init__(
        self, tokenizer, config: Config, skip_when_all_empty: List[str] = None
    ) -> None:
        super(DropReader, self).__init__(
            tokenizer,
            config,
            normalize_answer_function=_normalize_answer,
            skip_when_all_empty=skip_when_all_empty,
        )
        for item in self.skip_when_all_empty:
            assert item in [
                "passage_span",
                "question_span",
                "addition_subtraction",
                "counting",
                "multi_span",
            ], f"Unsupported skip type: {item}"

    def _read(self, file_path: str):
        print("Reading file at", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances, skip_count, cuts = [], 0, 0
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"].strip()
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                instance, is_cut = self.text_to_instance(
                    question_text=question_text,
                    passage_text=passage_text,
                    question_id=question_id,
                    passage_id=passage_id,
                    answer_annotations=answer_annotations,
                )

                cuts += is_cut
                if instance is not None:
                    instances.append(instance)
                else:
                    skip_count += 1

        print(f"Skipped {skip_count} questions, kept {len(instances)} questions.")
        print(f"Cut {cuts} samples.")
        return instances

    def text_to_instance(
        self,
        question_text: str,
        passage_text: str,
        question_id: str,
        passage_id: str,
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
            question_text,
            [passage_text],
            table=None,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            answer_type, answer_texts = self.extract_answer_info_from_annotation(
                answer_annotations[0]
            )

        specific_answer_type_is_multispan = False
        if answer_type in ["span", "spans"]:
            answer_texts = list(OrderedDict.fromkeys(answer_texts))
            if answer_type == "spans" and len(answer_texts) > 1:
                specific_answer_type_is_multispan = True

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

        # Adding a dummy number with the value 100 to handle percentage questions.
        passage_numbers.append(100)
        passage_number_indices.append(-1)
        passage_number_order.append(-1)

        passage_number_order = np.array(passage_number_order)
        question_number_order = np.array(question_number_order)

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

        target_numbers = []
        # `answer_texts` is a list of valid answers.
        for answer_text in answer_texts:
            number = get_number_from_text(answer_text)
            if number is not None:
                target_numbers.append(number)

        valid_signs_for_add_sub_expressions: List[List[int]] = []
        if answer_type in ["number", "date"]:
            target_number_strs = [
                convert_number_to_str(num, self.config) for num in target_numbers
            ]
            valid_signs_for_add_sub_expressions = find_valid_add_sub_expressions(
                passage_numbers, target_number_strs, config=self.config
            )

        valid_counts: List[int] = []
        if answer_type in ["number"]:
            # We support counting numbers from 0 to 9.
            numbers_for_count = list(range(10))
            valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

        # Add multi_span answer extraction.
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
                    # Possible if the passage was clipped, but not for all of the answers.
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

        valid_passage_spans = (
            valid_passage_spans
            if (not specific_answer_type_is_multispan or len(multi_span) < 1)
            and not self.config.no_single_span
            else []
        )
        valid_question_spans = (
            valid_question_spans
            if (not specific_answer_type_is_multispan or len(multi_span) < 1)
            and not self.config.no_single_span
            else []
        )

        type_to_answer_map = {
            "passage_span": valid_passage_spans,
            "question_span": valid_question_spans,
            "addition_subtraction": valid_signs_for_add_sub_expressions,
            "counting": valid_counts,
            "multi_span": multi_span,
        }

        if self.skip_when_all_empty and not any(
            type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty
        ):
            return None, is_cut

        if answer_annotations:
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
            elif valid_counts:
                answer_annotations[0]["answer_type_processed"] = AnswerOption.COUNTING
            elif multi_span:
                answer_annotations[0]["answer_type_processed"] = AnswerOption.MULTI_SPAN

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
            "passage_id": passage_id,
            "question_id": question_id,
            "answer_annotations": answer_annotations,
            "answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
            "answer_passage_spans": valid_passage_spans,
            "answer_question_spans": valid_question_spans,
            "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions,
            "counts": valid_counts,
            "multi_span": multi_span,
            "is_counts_multi_span": False,
        }

        return metadata, is_cut

    @staticmethod
    def extract_answer_info_from_annotation(
        answer_annotation: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = (
            answer_annotation[answer_type] if answer_type is not None else None
        )

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [
                answer_content[key]
                for key in ["month", "day", "year"]
                if key in answer_content and answer_content[key]
            ]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices
