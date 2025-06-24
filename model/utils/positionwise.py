from model.utils.kan import KANLinear
from model.utils.config import KANBertConfig
import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    Implementation of the PositionWiseFeedForward with KANLinear layers.
    """

    def __init__(self,
                 config: KANBertConfig,
                 activation: nn.Module = nn.SiLU(),
                 dropout: float = 0.1) -> None:
        """
        Constructor for the PositionWiseFeedForward class.

        Args:
            config (KANBertConfig): Configuration object for the KANBert model.
            activation (nn.Module): Activation function to be used in
                                    the feed forward layer.
            dropout (float): Dropout probability.
        """

        super().__init__()

        self.l1 = KANLinear(config.hidden_dim, config.intermediate_dim)
        self.l2 = KANLinear(config.intermediate_dim, config.hidden_dim)
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
