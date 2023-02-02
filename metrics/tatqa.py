from copy import deepcopy
from typing import Set, Tuple, Union
from metrics.tatqa_utils import *
import numpy as np
from scipy.optimize import linear_sum_assignment
from metrics.common import correct_pred_op
from model.utils import ANSWER_OPTION_TO_STR, AnswerOption

from tools.config import Config


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]],
    gold: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def extract_gold_answers(qa_annotation):
    """
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    """
    answer_type, scale = qa_annotation["answer_type"], qa_annotation["scale"]
    answer_content = qa_annotation["answer"]
    gold_answers = []
    if answer_type in ["multi-span", "span"]:  # list
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content  # multi-span
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ["count"]:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0, 0
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            else:
                if (
                    "%" in ans_str
                ):  #  has been handled the answer itself is a percentage
                    ans_str = "%.4f" % ans_num
                else:
                    ans_str = "%.4f" % (round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + " " + str(scale)
        ans_temp.append(ans_str)
    return [" ".join(ans_temp)]


# handle percentage
def add_percent_pred(prediction_strings, pred_scale, pred):
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    if (
        not pred_scale and "%" not in pred_str and is_number(pred_str)
    ):  # mode only or no pred_scale num only
        pred_str = to_number(pred_str)
        if pred_str is None:
            return prediction_strings
        prediction_strings.append("%.4f" % pred_str)
    return prediction_strings


class TaTQAEmAndF1(object):
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
            AnswerOption.AVERAGE: 0,
            AnswerOption.CHANGE_RATIO: 0,
            AnswerOption.DIVISION: 0,
            AnswerOption.COUNTING: 0,
            AnswerOption.MULTI_SPAN: 0,
            AnswerOption.COUNTING_AS_MULTI_SPAN: 0,
        }

    def __init__(self, config: Config, is_train: bool) -> None:
        self.config = config
        self.is_train = is_train

        self._total_count = 0
        self._total_em = 0.0
        self._total_f1 = 0.0

        self._init_count = 0
        self._scale_em = 0.0
        self._op_em = 0.0

        self.op_tp = self.get_op_dict()
        self.op_fp = self.get_op_dict()
        self.op_fn = self.get_op_dict()
        self.op_em = self.get_op_dict()
        self.op_f1 = self.get_op_dict()

        # self.scale_correct_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        # self.scale_total_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
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
    ):  # type: ignore
        """
        :param ground_truth:
        :param prediction:
        :param pred_scale:
        :param pred_span:
        :param gold_span:
        :param pred_op:
        :param gold_op:
        :return:
        """
        pred_op = prediction["predicted_operation"]
        predicted_answer = prediction["predicted_answer"]
        pred_scale = prediction["predicted_scale"]

        exact_match = None
        f1_score = None

        if ground_truth:
            ground_truth = ground_truth[0]
            gold_type, gold_answer, gold_scale = extract_gold_answers(ground_truth)
            if not gold_answer:
                exact_match = 0
                f1_score = 0
            else:
                ground_truth_answer_strings = get_answer_str(gold_answer, gold_scale)

                predicted_answer = (
                    predicted_answer
                    if isinstance(predicted_answer, list)
                    else [predicted_answer]
                )
                prediction_strings = get_answer_str(predicted_answer, pred_scale)
                prediction_strings = add_percent_pred(
                    prediction_strings, pred_scale, predicted_answer
                )
                exact_match, f1_score = metric_max_over_ground_truths(
                    get_metrics, prediction_strings, ground_truth_answer_strings
                )
                if gold_type in ["arithmetic", "count"]:
                    """if gold type equals with arithmetic and count, set the f1_score == exact_match"""
                    f1_score = exact_match

            self._total_em += exact_match
            self._total_f1 += f1_score
            self._total_count += 1

            if "answer_type_processed" in ground_truth:
                self._init_count += 1
                gold_op = ground_truth["answer_type_processed"]
                pred_op = correct_pred_op(pred_op, gold_op)
                if pred_op == gold_op:
                    self._op_em += 1
                    self.op_tp[gold_op] += 1
                    self.op_em[gold_op] += exact_match
                    self.op_f1[gold_op] += f1_score
                else:
                    self.op_fn[gold_op] += 1
                    self.op_fp[pred_op] += 1
                if pred_scale == ground_truth["scale_processed"]:
                    self._scale_em += 1
        else:
            ground_truth = dict()

        prediction = deepcopy(prediction)
        ground_truth = deepcopy(ground_truth)
        prediction["predicted_operation"] = ANSWER_OPTION_TO_STR[
            prediction["predicted_operation"]
        ]
        if "answer_type_processed" in ground_truth:
            ground_truth["answer_type_processed"] = ANSWER_OPTION_TO_STR[
                ground_truth["answer_type_processed"]
            ]
        it = {
            "question_id": question_id,
            "question_text": question_text,
            "passage_text": passage_text,
            **ground_truth,
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

        scale_score = (
            self._scale_em / self._init_count if self._init_count > 0 else None
        )
        op_acc = self._op_em / self._init_count if self._init_count > 0 else None

        to_ret = {
            "em": exact_match,
            "f1": f1_score,
            "scale": scale_score,
            "op_acc": op_acc,
        }

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
        self._total_count = 0
        self._total_em = 0.0
        self._total_f1 = 0.0

        self._init_count = 0
        self._scale_em = 0.0
        self._op_em = 0.0

        self.op_tp = self.get_op_dict()
        self.op_fp = self.get_op_dict()
        self.op_fn = self.get_op_dict()
        self.op_em = self.get_op_dict()
        self.op_f1 = self.get_op_dict()

        self._details = []

    def __str__(self):
        return f"TaTQAEmAndF1(em={self._total_em}, f1={self._total_f1}, count={self._total_count})"
