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


class SpanExtractor(SpanExtractorBase):
    class Result:
        def __init__(self, start_log_probs, end_log_probs, best_span) -> None:
            self.start_log_probs = start_log_probs
            self.end_log_probs = end_log_probs
            self.best_span = best_span

    def __init__(self, config: Config) -> None:
        super(SpanExtractor, self).__init__()

        self.config = config

        self.proj_sequence_g = nn.ModuleList()
        for i in range(3):
            self.proj_sequence_g.append(
                FFNLayer(config.hidden_size, config.hidden_size, 1, config.dropout)
            )

        self.span_start_predictor = SpanBoundPredictor(config=config, out_hidden_size=1)
        self.span_end_predictor = SpanBoundPredictor(config=config, out_hidden_size=1)

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

        return SpanExtractor.get_span_with_masks(
            start_logits=sequence_span_start_logits,
            end_logits=sequence_span_end_logits,
            span_mask=span_mask,
            qp_mask=question_mask + passage_mask,
        )

    @staticmethod
    def calc_log_marginal_likelihood(answer_as_spans, prediction: Result):
        # Shape: (batch_size, # of answer spans)
        gold_span_starts = answer_as_spans[:, :, 0]
        gold_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (gold_span_starts != -1).long()
        clamped_gold_span_starts = util.replace_masked_values(
            gold_span_starts, gold_span_mask, 0
        )
        clamped_gold_span_ends = util.replace_masked_values(
            gold_span_ends, gold_span_mask, 0
        )
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = torch.gather(
            prediction.start_log_probs, 1, clamped_gold_span_starts
        )
        log_likelihood_for_span_ends = torch.gather(
            prediction.end_log_probs, 1, clamped_gold_span_ends
        )
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = (
            log_likelihood_for_span_starts + log_likelihood_for_span_ends
        )
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = replace_masked_values_with_big_negative_number(
            log_likelihood_for_spans, gold_span_mask.bool()
        )
        # Shape: (batch_size, )
        log_marginal_likelihood = util.logsumexp(log_likelihood_for_spans)
        return log_marginal_likelihood

    @staticmethod
    def get_best_span(
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
        in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
        directly import this function without the class.

        We call the inputs "logits" - they could either be unnormalized logits or normalized log
        probabilities.  A log_softmax operation is a constant shifting of the entire logit
        vector, so taking an argmax over either one gives the same result.
        """
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        valid_span_log_probs = span_log_probs + mask.log()

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)

        # (batch_size, )
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        # (batch_size, 2)
        return torch.stack([span_start_indices, span_end_indices], dim=-1)

    @staticmethod
    def get_span_with_masks(start_logits, end_logits, span_mask, qp_mask):
        start_log_probs = util.masked_log_softmax(start_logits, qp_mask)
        end_log_probs = util.masked_log_softmax(end_logits, qp_mask)

        # Info about the best question span prediction
        # Shape: (batch_size, topk, 2)
        best_span = SpanExtractor.get_best_span(
            start_log_probs, end_log_probs, span_mask
        )

        return SpanExtractor.Result(start_log_probs, end_log_probs, best_span)
