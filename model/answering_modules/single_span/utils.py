import torch
import torch.nn as nn

from tools.config import Config


class SpanBoundPredictor(nn.Module):
    def __init__(self, config: Config, out_hidden_size: int) -> None:
        super(SpanBoundPredictor, self).__init__()

        self.config = config
        in_hidden_size = config.hidden_size * (4 if config.use_heavy_postnet else 3)
        self.span_bound_predictor = nn.Linear(
            in_hidden_size, out_hidden_size, bias=False
        )
        self.out_hidden_size = out_hidden_size
        if out_hidden_size > 1:
            self.activation = nn.ReLU()

    def forward(self, sequence_outputs, first_hidden, second_hidden):
        first_hidden = first_hidden.unsqueeze(1)
        second_hidden = second_hidden.unsqueeze(1)

        if self.config.use_heavy_postnet:
            sequence_for_span_bound = torch.cat(
                [
                    sequence_outputs[0],
                    sequence_outputs[1],
                    sequence_outputs[0] * first_hidden,
                    sequence_outputs[1] * second_hidden,
                ],
                dim=2,
            )
        else:
            sequence_for_span_bound = torch.cat(
                [
                    sequence_outputs,
                    sequence_outputs * first_hidden,
                    sequence_outputs * second_hidden,
                ],
                dim=2,
            )
        sequence_span_bound_logits = self.span_bound_predictor(sequence_for_span_bound)

        if self.out_hidden_size == 1:
            return sequence_span_bound_logits.squeeze(-1)
        else:
            return self.activation(sequence_span_bound_logits)


class SpanExtractorBase(nn.Module):
    def __init__(self) -> None:
        super(SpanExtractorBase, self).__init__()

    @staticmethod
    def update_answer(answer_json, predictions, metadata, i, scale_mask) -> None:
        predicted_span = tuple(predictions.best_span[i].detach().cpu().numpy())
        metadata = metadata[i]

        if predicted_span[0] <= len(metadata["question_token_offsets"]):
            res_str = metadata["original_question"]
            offsets = metadata["question_token_offsets"]
            start = 1
        else:
            res_str = metadata["original_passage"]
            offsets = metadata["passage_token_offsets"]
            start = len(metadata["question_tokens"]) + 2

        if scale_mask is not None:
            scale_mask[i][predicted_span[0] : predicted_span[1] + 1] = 1
        start_offset = offsets[predicted_span[0] - start][0]
        end_offset = offsets[predicted_span[1] - start][1]
        predicted_answer = res_str[start_offset:end_offset]
        answer_json["value"] = predicted_answer
        answer_json["predicted_span"] = [(start_offset, end_offset)]
        answer_json["predicted_answer"] = predicted_answer
