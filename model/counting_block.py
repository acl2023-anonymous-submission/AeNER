import torch
import torch.nn as nn
from model.utils import FFNLayer
from tools import allennlp as util
from tools.config import Config
from model.utils import replace_masked_values_with_big_negative_number


class CountingResult:
    def __init__(self, log_probs, best_number) -> None:
        self.log_probs = log_probs
        self.best_number = best_number


def calc_log_marginal_likelihood(answer_as_counts, prediction: CountingResult):
    # Count answers are padded with label -1,
    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
    # Shape: (batch_size, # of count answers)
    gold_mask = (answer_as_counts != -1).long()
    # Shape: (batch_size, # of count answers)
    clamped_gold = util.replace_masked_values(answer_as_counts, gold_mask, 0)
    log_likelihood = torch.gather(prediction.log_probs, 1, clamped_gold)
    # For those padded spans, we set their log probabilities to be very small negative value
    log_likelihood = replace_masked_values_with_big_negative_number(
        log_likelihood, gold_mask.bool()
    )
    # Shape: (batch_size, )
    log_marginal_likelihood = util.logsumexp(log_likelihood)
    return log_marginal_likelihood


def update_answer(answer_json, best_count_number):
    predicted_count = int(best_count_number.detach().cpu().numpy())
    predicted_answer = str(predicted_count)
    answer_json["predicted_count"] = predicted_count
    answer_json["predicted_answer"] = predicted_answer


class CountingBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super(CountingBlock, self).__init__()

        in_hidden_size = config.hidden_size * (5 if config.use_heavy_postnet else 4)
        self.number_predictor = FFNLayer(
            in_hidden_size, config.hidden_size, 10, config.dropout
        )

    def forward(self, number_vector, passage_h2, question_h2, sequence_output):
        # Shape: (batch_size, 10)
        logits = self.number_predictor(
            torch.cat(
                [number_vector, passage_h2, question_h2, sequence_output[:, 0]], dim=1
            )
        )
        log_probs = torch.nn.functional.log_softmax(logits, -1)
        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_number = torch.argmax(log_probs, -1)
        return CountingResult(log_probs, best_number)
