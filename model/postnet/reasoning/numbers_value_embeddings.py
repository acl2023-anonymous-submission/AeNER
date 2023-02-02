import torch
from torch import Tensor, nn
from typing import Any, List

from tools.config import Config

from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

EPS = 1e-10
MIN_EXP = -8
MAX_EXP = 15
# Number of exponent magnitudes.
NUM_PROTOTYPE_BASED_EMBEDDINGS = MAX_EXP - MIN_EXP + 1
# Max number of numbers in the input.
NUM_RANKING_EMBEDDINGS = 1000


class DefaultNumberValueEmbedding(nn.Module):
    def __init__(
        self, config: Config, number_value_embeddings=10, hidden_size=None
    ) -> None:
        super(DefaultNumberValueEmbedding, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        if hidden_size is not None:
            self.hidden_size = hidden_size
        self.number_value_embeddings_power = number_value_embeddings
        if config.trainable_number_value_embeddings_grid:
            self.number_value_embeddings_power = nn.parameter.Parameter(
                torch.tensor(number_value_embeddings)
            )
            print(
                "Trainable number_value_embeddings_power starterd with",
                self.number_value_embeddings_power.item(),
            )
        self.num_embeddings = (
            config.number_embeddins_range[1] - config.number_embeddins_range[0]
        )
        self.value_embs = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.hidden_size
        )

    def forward(self, numbers):
        number_value_embedding = 0.0
        for i in range(
            self.config.number_embeddins_range[0], self.config.number_embeddins_range[1]
        ):
            pw = self.number_value_embeddings_power ** i
            numbers_coeff = (numbers % pw) / pw / self.num_embeddings
            numbers_coeff = numbers_coeff.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            emb_value = self.value_embs(
                torch.tensor(
                    i - self.config.number_embeddins_range[0], device=self.config.device
                )
            )
            number_value_embedding = number_value_embedding + numbers_coeff * emb_value
        return number_value_embedding


class CnnNumberValueEmbedding(nn.Module):
    def __init__(self) -> None:
        super(CnnNumberValueEmbedding, self).__init__()

        char_embedding_dim = 256
        ngram_filter_sizes = (2, 3, 4, 5)
        num_filters = 256

        char_encoder = CnnEncoder(
            embedding_dim=char_embedding_dim,
            num_filters=num_filters,
            ngram_filter_sizes=ngram_filter_sizes,
        )

        token_char_embedding = TokenCharactersEncoder(
            Embedding(embedding_dim=char_embedding_dim, num_embeddings=14), char_encoder
        )
        self.text_field_embedder = BasicTextFieldEmbedder(
            {"token_characters": token_char_embedding}
        )

    def forward(self, all_number_strs):
        return self.text_field_embedder(all_number_strs["numbers"])


class PrototypeBasedEmbedding(nn.Module):
    def __init__(self, output_dim: int, device, sigma: float = 0.5):
        super().__init__()
        assert output_dim % 4 == 0, "output_dim should be divisible by 4"
        self.exp_dim = output_dim // 4
        self.mantissa_dim = 3 * output_dim // 4
        self.sigma = sigma
        self.q_values = (
            20 / (self.mantissa_dim - 1) * torch.arange(self.mantissa_dim).to(device)
            - 10
        )
        self.embeddings = nn.Embedding(
            NUM_PROTOTYPE_BASED_EMBEDDINGS, embedding_dim=self.exp_dim
        )

    def forward(self, numbers: Tensor) -> Tensor:
        exp_bases = torch.floor(torch.log10(numbers + EPS))
        exp_indexes = torch.clamp(
            exp_bases.int() - MIN_EXP, 0, NUM_PROTOTYPE_BASED_EMBEDDINGS - 1
        )
        mantissa = numbers.div(10 ** exp_bases)
        mantissa_embedding = torch.exp(
            -(((mantissa[:, :, None] - self.q_values[None, :]) / self.sigma) ** 2)
        )
        exp_embedding = self.embeddings(exp_indexes)
        return torch.cat([exp_embedding, mantissa_embedding], -1)


class PeriodicEmbedding(nn.Module):
    def __init__(self, output_dim: int, sigma: float = 0.5):
        super().__init__()
        assert output_dim % 2 == 0, "output_dim should be even"
        coefficients = torch.normal(0.0, sigma, (output_dim // 2,))
        self.coefficients = nn.Parameter(coefficients)

    def _cos_sin(self, x: Tensor) -> Tensor:
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

    def forward(self, numbers: Tensor) -> Tensor:
        return self._cos_sin(2 * torch.pi * self.coefficients * numbers[..., None])


class RankingEmbedding(nn.Module):
    def __init__(
        self, output_dim: int, num_ranking_embeddings: int = NUM_RANKING_EMBEDDINGS
    ):
        super().__init__()
        self.embeddings = nn.Embedding(num_ranking_embeddings, embedding_dim=output_dim)

    def forward(self, numbers: Tensor) -> Tensor:
        sorted_indexes = torch.argsort(numbers).int()
        return self.embeddings(sorted_indexes)


class EnsembleEmbedding(nn.Module):
    def __init__(self, numerical_embedders: List[nn.Module]):
        super().__init__()
        self.numerical_embedders = nn.ModuleList()
        for embedder in numerical_embedders:
            self.numerical_embedders.append(embedder)

    def forward(self, numbers: Tensor) -> Tensor:
        ensemble_output = [embedder(numbers) for embedder in self.numerical_embedders]
        return torch.cat(ensemble_output, dim=-1)
