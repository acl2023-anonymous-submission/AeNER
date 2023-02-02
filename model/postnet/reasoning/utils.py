import torch
import torch.nn as nn

from allennlp.nn.util import masked_mean

from tools.config import Config


class RelativeLoss(nn.Module):
    def __init__(self) -> None:
        super(RelativeLoss, self).__init__()

        self.cosine_sim = nn.CosineSimilarity(dim=3)

    def forward(
        self,
        embeds: torch.FloatTensor,
        ground_truth: torch.LongTensor,
        number_mask: torch.BoolTensor,
    ):
        bs = embeds.size(0)
        nsl = embeds.size(1)
        hs = embeds.size(2)

        embeds_cosine = (
            1.0
            - self.cosine_sim(
                embeds.unsqueeze(1).expand(bs, nsl, nsl, hs),
                embeds.unsqueeze(2).expand(bs, nsl, nsl, hs),
            ).flatten(start_dim=1)
        )

        number_mask = (
            (
                number_mask.unsqueeze(1).expand(bs, nsl, nsl)
                * number_mask.unsqueeze(2).expand(bs, nsl, nsl)
            )
            .triu(diagonal=1)
            .flatten(start_dim=1)
        )

        x = ground_truth.unsqueeze(1).expand(bs, nsl, nsl)
        y = ground_truth.unsqueeze(2).expand(bs, nsl, nsl)

        val_dist = (
            2 * torch.abs(x - y) / (torch.abs(x) + torch.abs(y) + 1e-18)
        ).flatten(start_dim=1)

        loss = torch.square(embeds_cosine - val_dist)
        loss = torch.sqrt(masked_mean(loss, number_mask, dim=1)).mean()

        return loss


class LogarithmLoss(nn.Module):
    def __init__(self, config: Config) -> None:
        super(LogarithmLoss, self).__init__()

        self.number_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1), nn.LeakyReLU()
        )
        self.number_decoder_loss = nn.L1Loss(reduce=False)

    def forward(
        self,
        embeds: torch.FloatTensor,
        ground_truth: torch.LongTensor,
        number_mask: torch.BoolTensor,
    ):
        decoded_numbers = self.number_decoder(embeds)
        number_value_embedding_loss = self.number_decoder_loss(
            decoded_numbers.squeeze(-1), (ground_truth + 1).log()
        )
        loss = masked_mean(number_value_embedding_loss, number_mask, dim=1).mean()

        return loss
