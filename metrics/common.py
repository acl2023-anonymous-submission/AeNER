from model.utils import AnswerOption


def correct_pred_op(pred_op: AnswerOption, gold_op: AnswerOption) -> bool:
    if pred_op == AnswerOption.SINGLE_SPAN and gold_op in [
        AnswerOption.QUESTION_SPAN,
        AnswerOption.SINGLE_SPAN,
    ]:
        return gold_op
    if (
        pred_op == AnswerOption.COUNTING
        and gold_op == AnswerOption.COUNTING_AS_MULTI_SPAN
    ):
        return gold_op
    if (
        pred_op == AnswerOption.COUNTING_AS_MULTI_SPAN
        and gold_op == AnswerOption.COUNTING
    ):
        return gold_op
    if pred_op == AnswerOption.SINGLE_SPAN:
        return AnswerOption.PASSAGE_SPAN
    return pred_op
