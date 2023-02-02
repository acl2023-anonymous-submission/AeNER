import torch
import torch.nn as nn
from model.utils import FFNLayer
from tools import allennlp as util
from tools.config import Config
from tools.utils import convert_number_to_str
from model.utils import replace_masked_values_with_big_negative_number


class ChangeRatioBlock(nn.Module):
    class Result:
        def __init__(self, log_probs, best_signs) -> None:
            self.log_probs = log_probs
            self.best_signs = best_signs

    def __init__(self, config: Config) -> None:
        super(ChangeRatioBlock, self).__init__()

        self.config = config
        in_hidden_size = config.hidden_size * (5 if config.use_heavy_postnet else 4)
        self.number_sign_predictor = FFNLayer(
            in_hidden_size, config.hidden_size, 3, config.dropout
        )

    def forward(
        self,
        encoded_numbers,
        question_hidden,
        passage_hidden,
        sequence_output,
        number_mask,
    ):
        alg_encoded_numbers = torch.cat(
            [
                encoded_numbers,
                question_hidden.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                passage_hidden.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                sequence_output[:, 0]
                .unsqueeze(1)
                .repeat(1, encoded_numbers.size(1), 1),
            ],
            2,
        )

        # Shape: (batch_size, # of numbers in the passage, 3)
        logits = self.number_sign_predictor(alg_encoded_numbers)
        log_probs = torch.nn.functional.log_softmax(logits, -1)

        # Shape: (batch_size, # of numbers in passage).
        best_signs = torch.argmax(log_probs, -1)
        # For padding numbers, the best sign masked as 0 (not included).
        best_signs = util.replace_masked_values(best_signs, number_mask, -1)

        return ChangeRatioBlock.Result(log_probs, best_signs)

    def update_answer(self, answer_json, best_signs, log_probs, metadata, scale_mask):
        # Plus-minus combination answer.
        original_numbers = metadata["passage_numbers"]
        number_indices = metadata["passage_number_indices"]
        sign_remap = {0: 0, 1: 0, 2: 0, -1: -1}
        predicted_signs = [sign_remap[it] for it in best_signs.detach().cpu().numpy()]

        a_candidates = []
        b_candidates = []
        for i in range(log_probs.shape[0]):
            if predicted_signs[i] == -1:
                continue
            if original_numbers[i] != 0:
                a_candidates.append((-log_probs[i][1].item(), i))
            b_candidates.append((-log_probs[i][2].item(), i))
        a_candidates.sort()
        b_candidates.sort()

        a = None
        b = None
        if not a_candidates:
            a = -1
        elif len(a_candidates) == 1:
            a = a_candidates[0][1]
            b = b_candidates[0][1]
            if a == b:
                if len(b_candidates) == 1:
                    b = -1
                else:
                    b = b_candidates[1][1]
        else:
            a = a_candidates[0][1]
            b = b_candidates[0][1]
            if a == b:
                if a_candidates[0][0] < b_candidates[0][0]:
                    b = b_candidates[1][1]
                else:
                    a = a_candidates[1][1]

        if a == -1 or b == -1:
            print("Something went wrong in (b-a)/a")
            result = 0
        else:
            if scale_mask is not None:
                scale_mask[number_indices[a]] = 1
                scale_mask[number_indices[b]] = 1
            a = original_numbers[a]
            b = original_numbers[b]
            result = (b - a) / a * 100

        predicted_answer = convert_number_to_str(result, self.config)

        offsets = metadata["passage_token_offsets"]
        number_indices = metadata["passage_number_indices"]
        number_positions = [offsets[index] for index in number_indices]
        answer_json["numbers"] = []
        for offset, value, sign in zip(
            number_positions, original_numbers, predicted_signs
        ):
            answer_json["numbers"].append(
                {"span": offset, "value": value, "sign": sign}
            )
        answer_json["value"] = result
        answer_json["predicted_answer"] = predicted_answer

    def calc_log_marginal_likelihood(
        answer_as_change_ratio_expressions, prediction, number_mask
    ):
        # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
        # Shape: (batch_size, # of combinations)
        gold_change_ratio_mask = (
            answer_as_change_ratio_expressions.sum(-1) > 0
        ).float()
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        gold_change_ratio_signs = answer_as_change_ratio_expressions.transpose(1, 2)
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        log_likelihood = torch.gather(prediction.log_probs, 2, gold_change_ratio_signs)
        # the log likelihood of the masked positions should be 0
        # so that it will not affect the joint probability
        log_likelihood = util.replace_masked_values(
            log_likelihood, number_mask.unsqueeze(-1), 0
        )
        # Shape: (batch_size, # of combinations)
        log_likelihood = log_likelihood.sum(1)
        # For those padded combinations, we set their log probabilities to be very small negative value
        log_likelihood = replace_masked_values_with_big_negative_number(
            log_likelihood, gold_change_ratio_mask.bool()
        )
        # Shape: (batch_size, )
        log_marginal_likelihood = util.logsumexp(log_likelihood)
        return log_marginal_likelihood
