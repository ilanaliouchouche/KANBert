from model.utils.kan import KANLinear
from model.utils.config import KANBertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):

    def __init__(self,
                 config: KANBertConfig,
                 activation: nn.Module = nn.SiLU()
                 ) -> None:

        super().__init__()

        self.k = config.top_k_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                KANLinear(config.hidden_dim, config.intermediate_dim),
                activation,
                nn.Dropout(config.dropout),
                KANLinear(config.intermediate_dim, config.hidden_dim),
                nn.Dropout(config.dropout)) for _ in range(config.num_experts)
        ])

        self.gate = KANLinear(config.hidden_dim, config.num_experts)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        router = self.gate(x)
        router = F.softmax(router, dim=-1)
        k_scores, k_indices = torch.topk(router, self.k, dim=-1)

        k_scores = k_scores.view(-1, self.k)
        k_indices = k_indices.view(-1, self.k)
        x_flat = x.view(-1, x.size(-1))

        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (k_indices == i)
            rows, cols = mask.nonzero(as_tuple=True)
            x_candidates = x_flat[rows]
            out = expert(x_candidates) * k_scores[rows, cols].unsqueeze(-1)
            output[rows] += out

        return output.view(*x.size())
