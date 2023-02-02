import enum
from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import allennlp as util
from allennlp.nn.util import replace_masked_values, min_value_of_dtype


class AnswerOption(enum.Enum):
    SINGLE_SPAN = 1
    ADDITION_SUBTRACTION = 2
    AVERAGE = 3
    CHANGE_RATIO = 4
    DIVISION = 5
    COUNTING = 6
    MULTI_SPAN = 7
    COUNTING_AS_MULTI_SPAN = 8
    QUESTION_SPAN = 9
    PASSAGE_SPAN = 10


STR_TO_ANSWER_OPTION = {
    "single_span": AnswerOption.SINGLE_SPAN,
    "addition_subtraction": AnswerOption.ADDITION_SUBTRACTION,
    "average": AnswerOption.AVERAGE,
    "change_ratio": AnswerOption.CHANGE_RATIO,
    "division": AnswerOption.DIVISION,
    "counting": AnswerOption.COUNTING,
    "multi_span": AnswerOption.MULTI_SPAN,
    "counting_as_multi_span": AnswerOption.COUNTING_AS_MULTI_SPAN,
    "question_span": AnswerOption.QUESTION_SPAN,
    "passage_span": AnswerOption.PASSAGE_SPAN,
}
ANSWER_OPTION_TO_STR = {
    answer_option: answer_str
    for answer_str, answer_option in STR_TO_ANSWER_OPTION.items()
}


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    Replace the masked values in a tensor something really negative so that they won't
    affect a max operation.
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))


def get_encoded_numbers_from_full_sequence(number_indices, encoded_for_numbers):
    number_mask = (number_indices > -1).long()
    clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
    encoded_numbers = torch.gather(
        encoded_for_numbers,
        1,
        clamped_number_indices.unsqueeze(-1).expand(
            -1, -1, encoded_for_numbers.size(-1)
        ),
    )
    return number_mask, encoded_numbers


def get_full_sequence_from_encoded_numbers(
    target_shape, encoded_numbers, number_indices, number_mask
):
    number_full_vec_size = (target_shape[0], target_shape[1] + 1, target_shape[2])
    number_full_vec = torch.zeros(
        number_full_vec_size, dtype=torch.float, device=encoded_numbers.device
    )
    clamped_passage_number_indices = util.replace_masked_values(
        number_indices, number_mask, number_full_vec.size(1) - 1
    )
    number_full_vec.scatter_(
        1,
        clamped_passage_number_indices.unsqueeze(-1).expand(
            -1, -1, encoded_numbers.size(-1)
        ),
        encoded_numbers,
    )
    return number_full_vec[:, :-1, :]


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class FFNLayer(nn.Module):
    def __init__(
        self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True
    ):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class GCNLog(nn.Module):
    def __init__(self) -> None:
        super(GCNLog, self).__init__()

    def forward(self, b, a):
        return torch.log((b - a).abs() + 1)


class GCNTanh(nn.Module):
    def __init__(self) -> None:
        super(GCNTanh, self).__init__()
        self.T = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, a):
        return torch.tanh((b - a).abs() * self.T)


class GCN(nn.Module):
    def __init__(
        self,
        node_dim,
        gcn_diff_opt: Union[str, None],
        extra_factor_dim=0,
        iteration_steps=1,
    ):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self.gcn_diff_opt = gcn_diff_opt
        if gcn_diff_opt is not None:
            if gcn_diff_opt == "log":
                self.diff_process = GCNLog()
            elif gcn_diff_opt == "tanh":
                self.diff_process = GCNTanh()
            else:
                raise Exception("Unknown gcn diff option: {}".format(gcn_diff_opt))

        self._node_weight_fc = torch.nn.Linear(
            node_dim + extra_factor_dim, 1, bias=True
        )

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

        if gcn_diff_opt is not None:
            self._dd_node_fc_left_bias = torch.nn.Linear(node_dim, node_dim, bias=False)
            self._qq_node_fc_left_bias = torch.nn.Linear(node_dim, node_dim, bias=False)
            self._dq_node_fc_left_bias = torch.nn.Linear(node_dim, node_dim, bias=False)
            self._qd_node_fc_left_bias = torch.nn.Linear(node_dim, node_dim, bias=False)

            self._dd_node_fc_right_bias = torch.nn.Linear(
                node_dim, node_dim, bias=False
            )
            self._qq_node_fc_right_bias = torch.nn.Linear(
                node_dim, node_dim, bias=False
            )
            self._dq_node_fc_right_bias = torch.nn.Linear(
                node_dim, node_dim, bias=False
            )
            self._qd_node_fc_right_bias = torch.nn.Linear(
                node_dim, node_dim, bias=False
            )

    def forward(
        self,
        d_node,
        q_node,
        d_numbers,
        q_numbers,
        d_node_mask,
        q_node_mask,
        graph,
        extra_factor=None,
    ):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        if self.gcn_diff_opt is not None:
            bsz = d_node.size(0)
            dd_numbers_logdiff = self.diff_process(
                d_numbers.unsqueeze(2).expand(bsz, d_node_len, d_node_len),
                d_numbers.unsqueeze(1).expand(bsz, d_node_len, d_node_len),
            )
            dq_numbers_logdiff = self.diff_process(
                d_numbers.unsqueeze(2).expand(bsz, d_node_len, q_node_len),
                q_numbers.unsqueeze(1).expand(bsz, d_node_len, q_node_len),
            )
            qd_numbers_logdiff = self.diff_process(
                q_numbers.unsqueeze(2).expand(bsz, q_node_len, d_node_len),
                d_numbers.unsqueeze(1).expand(bsz, q_node_len, d_node_len),
            )
            qq_numbers_logdiff = self.diff_process(
                q_numbers.unsqueeze(2).expand(bsz, q_node_len, q_node_len),
                q_numbers.unsqueeze(1).expand(bsz, q_node_len, q_node_len),
            )

        diagmat = torch.diagflat(
            torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device)
        )
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(
            torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device)
        )
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])

        d_node_neighbor_num = (
            dd_graph_left.sum(-1)
            + dd_graph_right.sum(-1)
            + dq_graph_left.sum(-1)
            + dq_graph_right.sum(-1)
        )
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(
            d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1
        )

        q_node_neighbor_num = (
            qq_graph_left.sum(-1)
            + qq_graph_right.sum(-1)
            + qd_graph_left.sum(-1)
            + qd_graph_right.sum(-1)
        )
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(
            q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1
        )

        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(
                    self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))
                ).squeeze(-1)
                q_node_weight = torch.sigmoid(
                    self._node_weight_fc(torch.cat((q_node, extra_factor), dim=-1))
                ).squeeze(-1)

            all_d_weight.append(d_node_weight)
            all_q_weight.append(q_node_weight)

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            # left
            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1), dd_graph_left, 0
            )
            qd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1), qd_graph_left, 0
            )
            qq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1), qq_graph_left, 0
            )
            dq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1), dq_graph_left, 0
            )

            if self.gcn_diff_opt is not None:
                dd_node_info_bias = self._dd_node_fc_left_bias(d_node)
                qd_node_info_bias = self._qd_node_fc_left_bias(d_node)
                qq_node_info_bias = self._qq_node_fc_left_bias(q_node)
                dq_node_info_bias = self._dq_node_fc_left_bias(q_node)

                dd_node_weight_bias = dd_node_weight
                qd_node_weight_bias = qd_node_weight
                qq_node_weight_bias = qq_node_weight
                dq_node_weight_bias = dq_node_weight

                dd_node_weight = dd_node_weight * dd_numbers_logdiff
                qd_node_weight = qd_node_weight * qd_numbers_logdiff
                qq_node_weight = qq_node_weight * qq_numbers_logdiff
                dq_node_weight = dq_node_weight * dq_numbers_logdiff

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)

            if self.gcn_diff_opt is not None:
                dd_node_info_left = dd_node_info_left + torch.matmul(
                    dd_node_weight_bias, dd_node_info_bias
                )
                qd_node_info_left = qd_node_info_left + torch.matmul(
                    qd_node_weight_bias, qd_node_info_bias
                )
                qq_node_info_left = qq_node_info_left + torch.matmul(
                    qq_node_weight_bias, qq_node_info_bias
                )
                dq_node_info_left = dq_node_info_left + torch.matmul(
                    dq_node_weight_bias, dq_node_info_bias
                )

            # right
            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

            dd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1), dd_graph_right, 0
            )
            qd_node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1), qd_graph_right, 0
            )
            qq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1), qq_graph_right, 0
            )
            dq_node_weight = util.replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1), dq_graph_right, 0
            )

            if self.gcn_diff_opt is not None:
                dd_node_info_bias = self._dd_node_fc_right_bias(d_node)
                qd_node_info_bias = self._qd_node_fc_right_bias(d_node)
                qq_node_info_bias = self._qq_node_fc_right_bias(q_node)
                dq_node_info_bias = self._dq_node_fc_right_bias(q_node)

                dd_node_weight_bias = dd_node_weight
                qd_node_weight_bias = qd_node_weight
                qq_node_weight_bias = qq_node_weight
                dq_node_weight_bias = dq_node_weight

                dd_node_weight = dd_node_weight * dd_numbers_logdiff
                qd_node_weight = qd_node_weight * qd_numbers_logdiff
                qq_node_weight = qq_node_weight * qq_numbers_logdiff
                dq_node_weight = dq_node_weight * dq_numbers_logdiff

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)

            if self.gcn_diff_opt is not None:
                dd_node_info_right = dd_node_info_right + torch.matmul(
                    dd_node_weight_bias, dd_node_info_bias
                )
                qd_node_info_right = qd_node_info_right + torch.matmul(
                    qd_node_weight_bias, qd_node_info_bias
                )
                qq_node_info_right = qq_node_info_right + torch.matmul(
                    qq_node_weight_bias, qq_node_info_bias
                )
                dq_node_info_right = dq_node_info_right + torch.matmul(
                    dq_node_weight_bias, dq_node_info_bias
                )

            agg_d_node_info = (
                dd_node_info_left
                + dd_node_info_right
                + dq_node_info_left
                + dq_node_info_right
            ) / d_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (
                qq_node_info_left
                + qq_node_info_right
                + qd_node_info_left
                + qd_node_info_right
            ) / q_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)

        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]

        all_d_weight = torch.cat(all_d_weight, dim=1)
        all_q_weight = torch.cat(all_q_weight, dim=1)

        return (
            d_node,
            q_node,
            all_d_weight,
            all_q_weight,
        )  # d_node_weight, q_node_weight


class AnswerAbilityPredictor(nn.Module):
    def __init__(self, hidden_size, dropout_prob, answering_abilities_num):
        super(AnswerAbilityPredictor, self).__init__()

        self.answer_ability_predictor = FFNLayer(
            3 * hidden_size, hidden_size, answering_abilities_num, dropout_prob
        )

    def forward(self, sequence_output, passage_h2, question_h2):
        # Shape: (batch_size, number_of_abilities)
        answer_ability_logits = self.answer_ability_predictor(
            torch.cat([passage_h2, question_h2, sequence_output[:, 0]], 1)
        )
        return F.log_softmax(answer_ability_logits, -1)
