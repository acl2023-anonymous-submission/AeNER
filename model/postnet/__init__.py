import torch
import torch.nn as nn
from model.utils import (
    ResidualGRU,
    get_encoded_numbers_from_full_sequence,
    get_full_sequence_from_encoded_numbers,
)
from model.postnet.utils import TensorMerger
from tools.config import Config


class PostNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super(PostNet, self).__init__()

        self.config = config

        if config.use_reasoning and not config.forced_reasoning:
            self.merger = TensorMerger(config=config, use_cat=False)

        if config.add_postnet_table_embeddings:
            self.table_y_embedding = nn.Embedding(
                config.table_y_embedding_len, config.hidden_size, padding_idx=0
            )
            self.table_x_embedding = nn.Embedding(
                config.table_x_embedding_len, config.hidden_size, padding_idx=0
            )

        if config.add_postnet_layernorm:
            self.proj_ln = nn.ModuleList()
            for _ in range(4):
                self.proj_ln.append(nn.LayerNorm(config.hidden_size))

        if config.postnet_transformer_layers > 0:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.bert_num_attention_heads,
                    dim_feedforward=4 * config.hidden_size,
                    dropout=config.bert_hidden_dropout_prob,
                    activation=config.bert_hidden_act,
                    layer_norm_eps=config.bert_layer_norm_eps,
                    batch_first=True,
                ),
                num_layers=config.postnet_transformer_layers,
            )

        if config.postnet_gru_layers > 0:
            self.gru = ResidualGRU(
                config.hidden_size, config.dropout, num_layers=config.postnet_gru_layers
            )

    def forward(
        self,
        sequence_output_or_list,
        gcn_info_vec,
        mask,
        table_position_ids,
        passage_number_indices: torch.LongTensor,
    ):
        if self.config.use_heavy_postnet:
            sequence_output_list = sequence_output_or_list
        else:
            sequence_output_list = [sequence_output_or_list]
        for i in range(len(sequence_output_list)):
            if self.config.use_reasoning:
                if self.config.forced_reasoning:
                    (
                        passage_number_mask,
                        encoded_numbers,
                    ) = get_encoded_numbers_from_full_sequence(
                        passage_number_indices, sequence_output_list[i]
                    )
                    sequence_output_list[i] = sequence_output_list[
                        i
                    ] - get_full_sequence_from_encoded_numbers(
                        sequence_output_list[i].size(),
                        encoded_numbers,
                        passage_number_indices,
                        passage_number_mask,
                    )
                    sequence_output_list[i] = sequence_output_list[i] + gcn_info_vec
                else:
                    sequence_output_list[i] = self.merger(
                        sequence_output_list[i], gcn_info_vec
                    )
            if self.config.add_postnet_table_embeddings:
                sequence_output_list[i] = sequence_output_list[i] + (
                    self.table_y_embedding(table_position_ids[:, 0, :])
                    + self.table_x_embedding(table_position_ids[:, 1, :])
                )
            if self.config.add_postnet_layernorm:
                sequence_output_list[i] = self.proj_ln[i](sequence_output_list[i])
            if self.config.postnet_transformer_layers > 0:
                sequence_output_list[i] = self.transformer(
                    sequence_output_list[i], src_key_padding_mask=mask
                )
            if self.config.postnet_gru_layers > 0:
                sequence_output_list[i] = self.gru(sequence_output_list[i])
        if self.config.use_heavy_postnet:
            return sequence_output_list
        else:
            return sequence_output_list[0]
