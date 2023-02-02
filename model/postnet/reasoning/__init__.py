import torch
import torch.nn as nn

from model.utils import (
    GCN,
    get_encoded_numbers_from_full_sequence,
    get_full_sequence_from_encoded_numbers,
)
from model.transformer_reasoner import TransformerReasoner

from tools.config import Config
from model.postnet.utils import TensorMerger
from model.postnet.reasoning.utils import LogarithmLoss, RelativeLoss
from model.postnet.reasoning.numbers_value_embeddings import (
    DefaultNumberValueEmbedding,
    CnnNumberValueEmbedding,
    PrototypeBasedEmbedding,
    PeriodicEmbedding,
    RankingEmbedding,
    EnsembleEmbedding,
)

NUMBER_EMBEDDING_TYPES = ["periodic", "prototype_based", "ranking", "ensemble"]


class ReasoningBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super(ReasoningBlock, self).__init__()

        self.config = config
        if config.use_heavy_postnet:
            self.gcn_input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        if config.number_value_embeddings is not None:
            self.merger = TensorMerger(config=config, use_cat=False)
            if isinstance(config.number_value_embeddings, float):
                self.number_value_embedder = DefaultNumberValueEmbedding(
                    config, config.number_value_embeddings
                )
            elif config.number_value_embeddings == "cnn":
                self.number_value_embedder = CnnNumberValueEmbedding()
            elif config.number_value_embeddings == "prototype_based":
                self.number_value_embedder = PrototypeBasedEmbedding(
                    config.hidden_size, config.device
                )
            elif config.number_value_embeddings == "periodic":
                self.number_value_embedder = PeriodicEmbedding(config.hidden_size)
            elif config.number_value_embeddings == "ranking":
                self.number_value_embedder = RankingEmbedding(config.hidden_size)
            elif config.number_value_embeddings == "ensemble":
                assert config.hidden_size % 4 == 0, "hidden size must be divisible by 4"
                output_dim_per_embedder = config.hidden_size // 4
                self.number_value_embedder = EnsembleEmbedding(
                    [
                        DefaultNumberValueEmbedding(
                            config, 10, output_dim_per_embedder
                        ),
                        PrototypeBasedEmbedding(output_dim_per_embedder, config.device),
                        PeriodicEmbedding(output_dim_per_embedder),
                        RankingEmbedding(output_dim_per_embedder),
                    ]
                )
            else:
                raise Exception(
                    "Unknown number_value_embeddings option: {}".format(
                        config.number_value_embeddings
                    )
                )
            if config.number_value_embeddings_loss is not None:
                if config.number_value_embeddings_loss == "logarithm":
                    self.num_embeddings_loss = LogarithmLoss(config)
                elif config.number_value_embeddings_loss == "relative":
                    self.num_embeddings_loss = RelativeLoss()
                else:
                    raise Exception(
                        "Check number_value_embeddings_loss {}".format(
                            config.number_value_embeddings_loss
                        )
                    )
        if config.reasoning_option == "gcn":
            self.reasoner = GCN(
                node_dim=config.hidden_size,
                iteration_steps=config.reasoning_steps,
                gcn_diff_opt=config.gcn_diff_opt,
            )
        elif config.reasoning_option in ["transformer"]:
            self.reasoner = TransformerReasoner(config=config)
        elif config.reasoning_option == "empty":
            print("Using empty reasoner.")
        else:
            raise "Unknown reasoning option: {}".format(config.reasoning_option)
        print("Reasoning iteration steps: %d" % config.reasoning_steps, flush=True)

    def forward(
        self,
        sequence_output_or_list,
        batch_size,
        passage_number_indices,
        question_number_indices,
        passage_number_order,
        question_number_order,
        passage_numbers,
        question_numbers,
        all_number_strs,
    ):
        if self.config.use_heavy_postnet:
            sequence_alg = self.gcn_input_proj(
                torch.cat(
                    [sequence_output_or_list[2], sequence_output_or_list[3]], dim=2
                )
            )
        else:
            sequence_alg = sequence_output_or_list

        (
            passage_number_mask,
            passage_encoded_numbers,
        ) = get_encoded_numbers_from_full_sequence(passage_number_indices, sequence_alg)
        (
            question_number_mask,
            question_encoded_numbers,
        ) = get_encoded_numbers_from_full_sequence(
            question_number_indices, sequence_alg
        )

        # Adding number value embeddings.
        number_value_embedding_loss = torch.tensor(0.0, device=self.config.device)
        if self.config.number_value_embeddings is not None:
            node = torch.cat(
                (question_encoded_numbers, passage_encoded_numbers), axis=1
            )
            numbers = torch.cat((question_numbers, passage_numbers), axis=1)

            if (
                isinstance(self.config.number_value_embeddings, float)
                or self.config.number_value_embeddings in NUMBER_EMBEDDING_TYPES
            ):
                number_value_embedding = self.number_value_embedder(numbers)
            elif self.config.number_value_embeddings == "cnn":
                number_value_embedding = self.number_value_embedder(all_number_strs)
            else:
                raise Exception(
                    "Unknown number_value_embeddings option: {}".format(
                        self.config.number_value_embeddings
                    )
                )

            node = self.merger(node, number_value_embedding)
            if self.config.number_value_embeddings_loss is not None:
                number_mask = (
                    torch.cat((question_number_indices, passage_number_indices), axis=1)
                    > -1
                )
                if self.config.number_value_embeddings_loss_after_bert:
                    embeds_for_loss = node
                else:
                    embeds_for_loss = number_value_embedding
                number_value_embedding_loss = self.num_embeddings_loss(
                    embeds=embeds_for_loss,
                    ground_truth=numbers,
                    number_mask=number_mask,
                )
            q_len = question_encoded_numbers.shape[1]
            question_encoded_numbers = node[:, :q_len, :]
            passage_encoded_numbers = node[:, q_len:, :]

        # graph mask
        new_graph_mask = None
        if self.config.reasoning_option == "gcn":
            number_order = torch.cat((passage_number_order, question_number_order), -1)
            new_graph_mask = number_order.unsqueeze(1).expand(
                batch_size, number_order.size(-1), -1
            ) > number_order.unsqueeze(-1).expand(batch_size, -1, number_order.size(-1))
            new_graph_mask = new_graph_mask.long()
            all_number_mask = torch.cat(
                (passage_number_mask, question_number_mask), dim=-1
            )
            new_graph_mask = (
                all_number_mask.unsqueeze(1)
                * all_number_mask.unsqueeze(-1)
                * new_graph_mask
            )
            del all_number_mask, number_order

        # iteration
        if self.config.reasoning_option != "empty":
            d_node, _, _, _ = self.reasoner(
                d_node=passage_encoded_numbers,
                q_node=question_encoded_numbers,
                d_numbers=passage_numbers,
                q_numbers=question_numbers,
                d_node_mask=passage_number_mask,
                q_node_mask=question_number_mask,
                graph=new_graph_mask,
            )
        else:
            d_node = passage_encoded_numbers

        return (
            get_full_sequence_from_encoded_numbers(
                sequence_alg.size(), d_node, passage_number_indices, passage_number_mask
            ),
            number_value_embedding_loss,
        )
