import torch
import torch.nn as nn

from tools.config import Config


class TransformerReasoner(nn.Module):
    def __init__(self, config: Config) -> None:
        super(TransformerReasoner, self).__init__()

        self.config = config

        if self.config.reasoning_steps > 0:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.bert_num_attention_heads,
                    dim_feedforward=4 * config.hidden_size,
                    dropout=config.bert_hidden_dropout_prob,
                    activation=config.bert_hidden_act
                    if (config.bert_hidden_act != "gelu_new")
                    else "gelu",
                    layer_norm_eps=config.bert_layer_norm_eps,
                    batch_first=True,
                ),
                num_layers=config.reasoning_steps,
            )

    def forward(
        self, d_node, q_node, d_numbers, q_numbers, d_node_mask, q_node_mask, graph
    ):
        len_d = d_node.shape[1]

        node = torch.cat((d_node, q_node), axis=1)
        mask = torch.cat((d_node_mask, q_node_mask), axis=1)

        if self.config.reasoning_steps > 0:
            node = self.transformer(node, src_key_padding_mask=mask)

        return (node[:, :len_d, :], node[:, len_d:, :], None, None)
