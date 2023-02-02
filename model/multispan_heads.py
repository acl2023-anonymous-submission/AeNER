import numpy as np
import torch
from torch.nn import Module

from tools.allennlp import replace_masked_values, logsumexp
from model.utils import replace_masked_values_with_big_negative_number


"""
From https://github.com/eladsegal/project-NLP-AML
"""


class MultiSpanHead(Module):
    def __init__(
        self, bert_dim: int, predictor: Module = None, dropout: float = 0.1
    ) -> None:
        super(MultiSpanHead, self).__init__()
        self.bert_dim = bert_dim
        self.dropout = dropout
        self.predictor = predictor or MultiSpanHead.default_predictor(
            self.bert_dim, self.dropout
        )

    def module(self):
        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError

    def prediction(self):
        raise NotImplementedError

    @staticmethod
    def default_predictor(bert_dim, dropout):
        return ff(bert_dim, bert_dim, 3, dropout)

    @staticmethod
    def decode_spans_from_tags(
        tags, question_passage_tokens, passage_text, question_text, wordpiece_mask
    ):
        spans_tokens = []

        prev = 0  # 0 = O
        current_tokens = []
        current_token_indices = []
        is_question = True

        for i, token in enumerate(question_passage_tokens):
            sep_token = token.text in ["</s>", "[SEP]"]

            if wordpiece_mask[i] == 0:
                if prev != 0:
                    current_tokens.append(token)
                    current_token_indices.append(i)
                continue

            if tags[i] == 1 and not sep_token:  # 1 = B
                if prev != 0:
                    spans_tokens.append(
                        (is_question, current_tokens, current_token_indices)
                    )
                    current_tokens = []
                    current_token_indices = []

                current_tokens.append(token)
                current_token_indices.append(i)
                prev = 1
                continue

            if tags[i] == 2 and not sep_token:  # 2 = I
                if prev != 0:
                    current_tokens.append(token)
                    current_token_indices.append(i)
                    prev = 2
                else:
                    # Illegal I, treat it as 0
                    prev = 0

            if (tags[i] == 0 or sep_token) and prev != 0:
                spans_tokens.append(
                    (is_question, current_tokens, current_token_indices)
                )
                current_tokens = []
                current_token_indices = []
                prev = 0

            if sep_token:
                is_question = False

        if current_tokens:
            spans_tokens.append((is_question, current_tokens, current_token_indices))

        spans_tokens = validate_tokens_spans(spans_tokens)
        spans_text, spans_indices, token_indices = decode_token_spans(
            spans_tokens, passage_text, question_text
        )

        return spans_text, spans_indices, token_indices


class FlexibleLoss(MultiSpanHead):
    def __init__(
        self, bert_dim: int, predictor: Module = None, dropout_prob: float = 0.1
    ) -> None:
        super(FlexibleLoss, self).__init__(bert_dim, predictor, dropout_prob)

    def module(self, bert_out, seq_mask=None):
        logits = self.predictor(bert_out)

        if seq_mask is not None:
            log_probs = replace_masked_values(
                torch.nn.functional.log_softmax(logits, dim=-1),
                seq_mask.unsqueeze(-1),
                0.0,
            )
            logits = replace_masked_values_with_big_negative_number(
                logits, seq_mask.unsqueeze(-1).bool()
            )
        else:
            log_probs = torch.nn.functional.log_softmax(logits)

        return log_probs, logits

    def log_likelihood(
        self,
        answer_as_text_to_disjoint_bios,
        answer_as_list_of_bios,
        span_bio_labels,
        log_probs,
        seq_mask,
        is_bio_mask,
    ):
        # answer_as_text_to_disjoint_bios - Shape: (batch_size, # of text answers, # of spans a for text answer, seq_length)
        # answer_as_list_of_bios - Shape: (batch_size, # of correct sequences, seq_length)
        # log_probs - Shape: (batch_size, seq_length, 3)
        # seq_mask - Shape: (batch_size, seq_length)

        # Generate most likely correct predictions
        with torch.no_grad():
            answer_as_list_of_bios = answer_as_list_of_bios * seq_mask.unsqueeze(1)
            if answer_as_text_to_disjoint_bios.sum() > 0:
                full_bio = span_bio_labels

                is_pregenerated_answer_format_mask = (
                    answer_as_list_of_bios.sum((1, 2)) > 0
                ).long()
                list_of_bios = torch.cat(
                    (
                        answer_as_list_of_bios,
                        (
                            full_bio
                            * (1 - is_pregenerated_answer_format_mask).unsqueeze(-1)
                        ).unsqueeze(1),
                    ),
                    dim=1,
                )
            else:
                list_of_bios = answer_as_list_of_bios

        ### Calculate log-likelihood from list_of_bios
        log_marginal_likelihood_for_multispan = self._get_combined_likelihood(
            list_of_bios, log_probs
        )

        # For questions without spans, we set their log probabilities to be very small negative value
        log_marginal_likelihood_for_multispan = (
            replace_masked_values_with_big_negative_number(
                log_marginal_likelihood_for_multispan, is_bio_mask.bool()
            )
        )

        return log_marginal_likelihood_for_multispan

    def prediction(self, logits, qp_tokens, p_text, q_text, seq_mask, wordpiece_mask):
        predicted_tags = torch.argmax(logits, dim=-1)
        predicted_tags = replace_masked_values(predicted_tags, seq_mask, 0)

        return MultiSpanHead.decode_spans_from_tags(
            predicted_tags, qp_tokens, p_text, q_text, wordpiece_mask
        )

    def _get_combined_likelihood(self, answer_as_list_of_bios, log_probs):
        # answer_as_list_of_bios - Shape: (batch_size, # of correct sequences, seq_length)
        # log_probs - Shape: (batch_size, seq_length, 3)

        # Shape: (batch_size, # of correct sequences, seq_length, 3)
        # duplicate log_probs for each gold bios sequence
        expanded_log_probs = log_probs.unsqueeze(1).expand(
            -1, answer_as_list_of_bios.size()[1], -1, -1
        )

        # get the log-likelihood per each sequence index
        # Shape: (batch_size, # of correct sequences, seq_length)
        log_likelihoods = torch.gather(
            expanded_log_probs, dim=-1, index=answer_as_list_of_bios.unsqueeze(-1)
        ).squeeze(-1)

        # Shape: (batch_size, # of correct sequences)
        correct_sequences_pad_mask = (answer_as_list_of_bios.sum(-1) > 0).long()

        # Sum the log-likelihoods for each index to get the log-likelihood of the sequence
        # Shape: (batch_size, # of correct sequences)
        sequences_log_likelihoods = log_likelihoods.sum(dim=-1)
        sequences_log_likelihoods = replace_masked_values_with_big_negative_number(
            sequences_log_likelihoods, correct_sequences_pad_mask.bool()
        )

        # Sum the log-likelihoods for each sequence to get the marginalized log-likelihood over the correct answers
        log_marginal_likelihood = logsumexp(sequences_log_likelihoods, dim=-1)

        return log_marginal_likelihood


def ff(input_dim, hidden_dim, output_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def validate_tokens_spans(spans_tokens):
    valid_tokens = []
    for is_question, tokens, token_indices in spans_tokens:
        tokens_text = [token.text for token in tokens]
        if not ("<s>" in tokens_text or "</s>" in tokens_text):
            valid_tokens.append((is_question, tokens, token_indices))

    return valid_tokens


def decode_token_spans(spans_tokens, passage_text, question_text):
    spans_text = []
    spans_indices = []
    token_bounds_indices = []

    for is_question, tokens, token_indices in spans_tokens:
        token_bounds_indices.append((token_indices[0], token_indices[-1] + 1))

        text_start = tokens[0].idx
        text_end = tokens[-1].edx

        spans_indices.append(
            ("question" if is_question else "passage", text_start, text_end)
        )

        if is_question:
            spans_text.append(question_text[text_start:text_end])
        else:
            spans_text.append(passage_text[text_start:text_end])

    return spans_text, spans_indices, token_bounds_indices


def remove_substring_from_prediction(spans):
    new_spans = []
    lspans = [s.lower() for s in spans]

    for span in spans:
        lspan = span.lower()

        if lspans.count(lspan) > 1:
            lspans.remove(lspan)
            continue

        if not any(
            (
                lspan + " " in s
                or " " + lspan in s
                or lspan + "s" in s
                or lspan + "n" in s
                or (lspan in s and not s.startswith(lspan) and not s.endswith(lspan))
            )
            and lspan != s
            for s in lspans
        ):
            new_spans.append(span)

    return new_spans
