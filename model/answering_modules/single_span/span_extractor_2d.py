import torch
import torch.nn as nn
from model.utils import FFNLayer
from tools import allennlp as util
from tools.config import Config
from model.utils import replace_masked_values_with_big_negative_number
from model.answering_modules.single_span.utils import (
    SpanExtractorBase,
    SpanBoundPredictor,
)


class SpanExtractor2D(SpanExtractorBase):
    class Result:
        def __init__(self, log_probs, best_span) -> None:
            self.log_probs = log_probs
            self.best_span = best_span

    def __init__(self, config: Config) -> None:
        super(SpanExtractor2D, self).__init__()

        self.config = config

        self.proj_sequence_g = nn.ModuleList()
        for _ in range(3):
            self.proj_sequence_g.append(
                FFNLayer(config.hidden_size, config.hidden_size, 1, config.dropout)
            )

        self.span_start_predictor = SpanBoundPredictor(
            config=config, out_hidden_size=config.hidden_size // 2
        )
        self.span_end_predictor = SpanBoundPredictor(
            config=config, out_hidden_size=config.hidden_size // 2
        )
        self.net = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
        )

    def forward(self, sequence_output_or_list, question_mask, passage_mask, span_mask):
        # passage g0, g1, g2
        question_g = []
        for i in range(3):
            if self.config.use_heavy_postnet:
                question_g_weight = self.proj_sequence_g[i](
                    sequence_output_or_list[i]
                ).squeeze(-1)
                question_g_weight = util.masked_softmax(
                    question_g_weight, question_mask
                )
                question_g.append(
                    util.weighted_sum(sequence_output_or_list[i], question_g_weight)
                )
            else:
                question_g_weight = self.proj_sequence_g[i](
                    sequence_output_or_list
                ).squeeze(-1)
                question_g_weight = util.masked_softmax(
                    question_g_weight, question_mask
                )
                question_g.append(
                    util.weighted_sum(sequence_output_or_list, question_g_weight)
                )

        if self.config.use_heavy_postnet:
            sequence_span_start_logits = self.span_start_predictor(
                [sequence_output_or_list[2], sequence_output_or_list[0]],
                question_g[2],
                question_g[0],
            )
            sequence_span_end_logits = self.span_end_predictor(
                [sequence_output_or_list[2], sequence_output_or_list[1]],
                question_g[2],
                question_g[1],
            )
        else:
            sequence_span_start_logits = self.span_start_predictor(
                sequence_output_or_list, question_g[2], question_g[0]
            )
            sequence_span_end_logits = self.span_end_predictor(
                sequence_output_or_list, question_g[2], question_g[1]
            )

        bs, sl, hs = sequence_span_start_logits.size()

        sequence_span_start_logits = sequence_span_start_logits.unsqueeze(2).expand(
            bs, sl, sl, hs
        )
        sequence_span_end_logits = sequence_span_end_logits.unsqueeze(1).expand(
            bs, sl, sl, hs
        )

        span_logits = torch.cat(
            (sequence_span_start_logits, sequence_span_end_logits), dim=3
        )

        span_logits = self.net(span_logits).squeeze(-1)

        return SpanExtractor2D.get_span_with_masks(
            span_logits=span_logits, span_mask=span_mask
        )

    @staticmethod
    def calc_log_marginal_likelihood(answer_as_spans, prediction: Result):
        batch_size, seq_len, _ = prediction.log_probs.size()

        # Shape: (batch_size, # of answer spans)
        gold_span_poses = answer_as_spans[:, :, 0] * seq_len + answer_as_spans[:, :, 1]

        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (answer_as_spans[:, :, 0] != -1).long()
        clamped_gold_span_poses = util.replace_masked_values(
            gold_span_poses, gold_span_mask, 0
        )
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = torch.gather(
            prediction.log_probs.flatten(start_dim=1), 1, clamped_gold_span_poses
        )
        # Shape: (batch_size, # of answer spans)
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = replace_masked_values_with_big_negative_number(
            log_likelihood_for_spans, gold_span_mask.bool()
        )
        # Shape: (batch_size, )
        log_marginal_likelihood = util.logsumexp(log_likelihood_for_spans)
        return log_marginal_likelihood

    @staticmethod
    def get_best_span(log_probs: torch.Tensor, passage_length: int) -> torch.Tensor:
        best_spans = log_probs.argmax(-1)

        # (batch_size, )
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        # (batch_size, 2)
        return torch.stack([span_start_indices, span_end_indices], dim=-1)

    @staticmethod
    def get_span_with_masks(span_logits, span_mask):
        seq_len = span_logits.size(1)
        span_mask = span_mask.flatten(start_dim=1)
        span_logits = span_logits.flatten(start_dim=1)
        log_probs = util.masked_log_softmax(span_logits, span_mask)

        best_span = SpanExtractor2D.get_best_span(log_probs, seq_len)

        log_probs = log_probs.view(-1, seq_len, seq_len)
        return SpanExtractor2D.Result(log_probs, best_span)
