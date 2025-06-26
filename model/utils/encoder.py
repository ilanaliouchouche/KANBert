from model.utils.multiheadattention import MultiHeadAttention
from model.utils.positionwise import PositionWiseFeedForward
from torch import nn
import torch
from model.utils.config import KANBertConfig


class Encoder(nn.Module):

    def __init__(self,
                 config: KANBertConfig) -> None:

        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.attention = MultiHeadAttention(config)

        self.ff = PositionWiseFeedForward(config)

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:

        x = self.norm1(x + self.attention(x, attention_mask))
        x = self.norm2(x + self.ff(x))
        return x
