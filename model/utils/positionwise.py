import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, 
                 intermediate_dim: int, 
                 hidden_dim: int,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.1) -> None:

        super().__init__()

        self.l1 = nn.Linear(intermediate_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, intermediate_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.l1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)

        return x