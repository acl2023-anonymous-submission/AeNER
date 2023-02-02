from copy import deepcopy
from typing import List, Tuple
from model.utils import ANSWER_OPTION_TO_STR, AnswerOption
from tools.utils import metric_max_over_ground_truths
from metrics.drop_utils import get_metrics as drop_em_and_f1
from metrics.drop_utils import answer_json_to_strings
from metrics.common import correct_pred_op

from tools.config import Config


class DropEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    @staticmethod
    def get_op_dict():
        return {
            AnswerOption.PASSAGE_SPAN: 0,
            AnswerOption.QUESTION_SPAN: 0,
            AnswerOption.ADDITION_SUBTRACTION: 0,
            AnswerOption.COUNTING: 0,
            AnswerOption.MULTI_SPAN: 0,
        }

    def __init__(self, config: Config, is_train: bool) -> None:
        self.config = config
        self.is_train = is_train

        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_count = 0

        self._init_count = 0
        self._op_em = 0.0

        self.op_tp = self.get_op_dict()
        self.op_fp = self.get_op_dict()
        self.op_fn = self.get_op_dict()
        self.op_em = self.get_op_dict()
        self.op_f1 = self.get_op_dict()

        self._details = []

    def __call__(
        self,
        prediction: dict,
        ground_truth: List[dict],
        question_id: str,
        question_text: str,
        passage_text: str,
        metas_logits: dict,
        idx_in_batch: int,
    ) -> None:
        pred_op = prediction["predicted_operation"]
        predicted_answer = prediction["predicted_answer"]

        exact_match = None
        f1_score = None

        if ground_truth:
            ground_truth_answer_strings = [
                answer_json_to_strings(annotation)[0] for annotation in ground_truth
            ]
            exact_match, f1_score = metric_max_over_ground_truths(
                drop_em_and_f1, predicted_answer, ground_truth_answer_strings
            )
            self._total_em += exact_match
            self._total_f1 += f1_score
            self._total_count += 1

            if "answer_type_processed" in ground_truth[0]:
                self._init_count += 1
                gold_op = ground_truth[0]["answer_type_processed"]
                pred_op = correct_pred_op(pred_op, gold_op)
                if pred_op == gold_op:
                    self._op_em += 1
                    self.op_tp[gold_op] += 1
                    self.op_em[gold_op] += exact_match
                    self.op_f1[gold_op] += f1_score
                else:
                    self.op_fn[gold_op] += 1
                    self.op_fp[pred_op] += 1
        else:
            ground_truth = [dict()]

        special_data = dict()
        if not self.is_train and self.config.full_logging:
            for key in metas_logits["metadata"][idx_in_batch]:
                if key in [
                    "answer_annotations",
                    "question_passage_tokens",
                    "question_number_order",
                    "passage_number_order",
                ]:
                    continue
                special_data[key] = metas_logits["metadata"][idx_in_batch][key]
            if "answer_ability_log_probs" in metas_logits["predictions"]:
                special_data["answer_ability_log_probs"] = metas_logits["predictions"][
                    "answer_ability_log_probs"
                ][idx_in_batch].tolist()
            if AnswerOption.SINGLE_SPAN in metas_logits["predictions"]:
                special_data["single_span_start_log_probs"] = (
                    metas_logits["predictions"][AnswerOption.SINGLE_SPAN]
                    .start_log_probs[idx_in_batch]
                    .tolist()
                )
                special_data["single_span_end_log_probs"] = (
                    metas_logits["predictions"][AnswerOption.SINGLE_SPAN]
                    .end_log_probs[idx_in_batch]
                    .tolist()
                )
            if AnswerOption.COUNTING in metas_logits["predictions"]:
                special_data["counting_log_probs"] = (
                    metas_logits["predictions"][AnswerOption.COUNTING]
                    .log_probs[idx_in_batch]
                    .tolist()
                )
            if AnswerOption.MULTI_SPAN in metas_logits["predictions"]:
                special_data["multi_span_log_probs"] = (
                    metas_logits["predictions"][AnswerOption.MULTI_SPAN]
                    .log_probs[idx_in_batch]
                    .tolist()
                )
            if AnswerOption.ADDITION_SUBTRACTION in metas_logits["predictions"]:
                special_data["add_sub_log_probs"] = (
                    metas_logits["predictions"][AnswerOption.ADDITION_SUBTRACTION]
                    .log_probs[idx_in_batch]
                    .tolist()
                )

        prediction = deepcopy(prediction)
        ground_truth = deepcopy(ground_truth)
        prediction["predicted_operation"] = ANSWER_OPTION_TO_STR[
            prediction["predicted_operation"]
        ]
        if "answer_type_processed" in ground_truth[0]:
            ground_truth[0]["answer_type_processed"] = ANSWER_OPTION_TO_STR[
                ground_truth[0]["answer_type_processed"]
            ]
        it = {
            "question_id": question_id,
            "question_text": question_text,
            "passage_text": passage_text,
            **ground_truth[0],
            **special_data,
            **prediction,
            "f1": f1_score,
            "em": exact_match,
        }
        self._details.append(it)

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = (
            self._total_em / self._total_count if self._total_count > 0 else None
        )
        f1_score = self._total_f1 / self._total_count if self._total_count > 0 else None

        op_acc = self._op_em / self._init_count if self._init_count > 0 else None

        to_ret = {"em": exact_match, "f1": f1_score, "op_acc": op_acc}

        for op in self.get_op_dict():
            to_ret[ANSWER_OPTION_TO_STR[op] + "/pr"] = (
                self.op_tp[op] / (self.op_tp[op] + self.op_fp[op])
                if (self.op_tp[op] + self.op_fp[op]) > 0
                else None
            )
            to_ret[ANSWER_OPTION_TO_STR[op] + "/rec"] = (
                self.op_tp[op] / (self.op_tp[op] + self.op_fn[op])
                if (self.op_tp[op] + self.op_fn[op]) > 0
                else None
            )
            to_ret[ANSWER_OPTION_TO_STR[op] + "/em"] = (
                self.op_em[op] / self.op_tp[op] if self.op_tp[op] > 0 else None
            )
            to_ret[ANSWER_OPTION_TO_STR[op] + "/f1"] = (
                self.op_f1[op] / self.op_tp[op] if self.op_tp[op] > 0 else None
            )

        if reset:
            self.reset()

        return to_ret

    def get_raw(self):
        return self._details

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_count = 0

        self._init_count = 0
        self._op_em = 0.0

        self.op_tp = self.get_op_dict()
        self.op_fp = self.get_op_dict()
        self.op_fn = self.get_op_dict()
        self.op_em = self.get_op_dict()
        self.op_f1 = self.get_op_dict()

        self._details = []

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1}, count={self._total_count})"
