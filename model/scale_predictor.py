import torch
import torch.nn as nn
from model.utils import FFNLayer
from tools import allennlp as util
from dataset.utils import ID_TO_ANSWER_SCALE
from tools.config import Config
from model.utils import replace_masked_values_with_big_negative_number


class ScaleBlock(nn.Module):
    class ScaleResult:
        def __init__(self, log_probs, best_scale) -> None:
            self.log_probs = log_probs
            self.best_scale = best_scale

    def __init__(self, config: Config) -> None:
        super(ScaleBlock, self).__init__()

        self.answer_depended_scale_predictor = config.answer_depended_scale_predictor

        hidden_multiplier = 4
        if config.use_heavy_postnet:
            hidden_multiplier += 1
        if config.answer_depended_scale_predictor:
            hidden_multiplier += 1
        self.number_predictor = FFNLayer(
            config.hidden_size * hidden_multiplier,
            config.hidden_size,
            len(ID_TO_ANSWER_SCALE),
            config.dropout,
        )

    def forward(
        self, number_vector, passage_h2, question_h2, answer_h2, sequence_output
    ):
        # Shape: (batch_size, number_of_scales)
        model_input = [number_vector, passage_h2, question_h2, sequence_output[:, 0]]
        if self.answer_depended_scale_predictor:
            model_input.append(answer_h2)
        logits = self.number_predictor(torch.cat(model_input, dim=1))
        log_probs = torch.nn.functional.log_softmax(logits, -1)
        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_scale = torch.argmax(log_probs, -1)
        return ScaleBlock.ScaleResult(log_probs, best_scale)

    @staticmethod
    def calc_log_marginal_likelihood(answer_scales, prediction: ScaleResult):
        # Count answers are padded with label -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        # Shape: (batch_size, # of count answers)
        gold_mask = (answer_scales != -1).long()
        # Shape: (batch_size, # of count answers)
        clamped_gold = util.replace_masked_values(answer_scales, gold_mask, 0)
        log_likelihood = torch.gather(prediction.log_probs, 1, clamped_gold)
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood = replace_masked_values_with_big_negative_number(
            log_likelihood, gold_mask.bool()
        )
        # Shape: (batch_size, )
        log_marginal_likelihood = util.logsumexp(log_likelihood)
        return log_marginal_likelihood

    @staticmethod
    def update_answer(answer_json, best_scale) -> None:
        predicted_scale = best_scale.item()
        predicted_scale = ID_TO_ANSWER_SCALE[predicted_scale]
        answer_json["predicted_scale"] = predicted_scale
