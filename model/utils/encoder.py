from model.utils.multiheadattention import MultiHeadAttention
from model.utils.positionwise import PositionWiseFeedForward
from torch import nn
import torch
from model.utils.config import KANBertConfig


class Encoder(nn.Module):
    """
    Implementation of the Encoder layer in the KANBert model.
    """

    def __init__(self,
                 config: KANBertConfig) -> None:
        """
        Constructor for the Encoder class.

        Args:
            config (KANBertConfig): Configuration object for the KANBert model.
        """

        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.attention = MultiHeadAttention(config)

        self.ff = PositionWiseFeedForward(config)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Encoder layer.

        Args:
            x (torch.Tensor): Input hidden states of shape
                              (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output hidden states of shape
                          (batch_size, seq_len, hidden_dim).
        """

        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x
