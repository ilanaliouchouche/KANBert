from .kan import KANLinear
from .rope import RoPE
from torch import nn
import torch
from .config import KANBertConfig

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 config: KANBertConfig) -> None:
        
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.hidden_dim // config.num_attention_heads

        self.q = KANLinear(config.hidden_dim, config.hidden_dim)
        self.k = KANLinear(config.hidden_dim, config.hidden_dim)
        self.v = KANLinear(config.hidden_dim, config.hidden_dim)

        self.out = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.position_encoding = RoPE(config)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, _ = x.size()

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.position_encoding(q, k)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_head_dim ** 0.5)
        attention_alphas = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_alphas, v)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)

        return self.out(attention_output)