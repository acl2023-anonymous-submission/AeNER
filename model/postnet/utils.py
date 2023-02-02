import torch
import torch.nn as nn

from tools.config import Config


class TensorMerger(nn.Module):
    def __init__(self, config: Config, use_cat: bool) -> None:
        super(TensorMerger, self).__init__()

        self.use_cat = use_cat
        if use_cat:
            self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, left, right):
        if not self.use_cat:
            return left + right
        catted = torch.cat([left, right], -1)
        return self.linear(catted)
