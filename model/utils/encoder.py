from .multiheadattention import MultiHeadAttention
from .positionwise import PositionWiseFeedForward
from torch import nn
import torch
from .config import KANBertConfig

class Encoder(nn.Module):

    def __init__(self,
                 config: KANBertConfig) -> None:
        
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.attention = MultiHeadAttention(config)
        
        self.ff = PositionWiseFeedForward(config)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x