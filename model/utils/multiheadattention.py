from .kan import KANLinear
from .rope import RoPE
 
from torch import nn
import torch

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_attention_heads: int,
                 max_sequence_len: int,
                 **kwargs) -> None:
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = hidden_dim // num_attention_heads

        self.q = KANLinear(hidden_dim, hidden_dim)
        self.k = KANLinear(hidden_dim, hidden_dim)
        self.v = KANLinear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.position_encoding = RoPE(hidden_dim,
                                      max_sequence_len,
                                      **kwargs)


    def forward(self,
                query: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, _ = query.size()

        q: torch.Tensor = self.q(query)
        k: torch.Tensor = self.k(query)
        v: torch.Tensor = self.v(query)

        q = q.view(batch_size,
                   seq_len,
                   self.num_attention_heads,
                   self.attention_head_dim) # (batch_size, seq_len, num_attention_heads, attention_head_dim)

        k = k.view(batch_size,
                   seq_len,
                   self.num_attention_heads,
                   self.attention_head_dim) # (batch_size, seq_len, num_attention_heads, attention_head_dim)
        
        v = v.view(batch_size,
                   seq_len,
                   self.num_attention_heads,
                   self.attention_head_dim) # (batch_size, seq_len, num_attention_heads, attention_head_dim)
        

        q = q.transpose(1, 2) # (batch_size, num_attention_heads, seq_len, attention_head_dim)
        k = k.transpose(1, 2) # (batch_size, num_attention_heads, seq_len, attention_head_dim)
        v = v.transpose(1, 2) # (batch_size, num_attention_heads, seq_len, attention_head_dim)


        attention_scores = (
            torch.matmul(q, k.transpose(-2, -1)) / 
            (self.attention_head_dim ** 0.5)
            ) # (batch_size, num_attention_heads, seq_len, seq_len)
        
        attention_alphas = torch.softmax(attention_scores, dim=-1) # (batch_size, num_attention_heads, seq_len, seq_len)

        attention_output = torch.matmul(attention_alphas, v) # (batch_size, num_attention_heads, seq_len, attention_head_dim)

        attention_output = attention_output.transpose(1, 2).contiguous() # (batch_size, seq_len, num_attention_heads, attention_head_dim)
        
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim) # (batch_size, seq_len, hidden_dim)

        return self.out(attention_output) # (batch_size, seq_len, hidden_dim)
        

