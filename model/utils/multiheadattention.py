from model.utils.kan import KANLinear
from model.utils.rope import RoPE
from torch import nn
import torch.nn.functional as F
import torch
import math
from model.utils.config import KANBertConfig


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 config: KANBertConfig) -> None:

        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_attention_heads = config.num_attention_heads
        self.qkv = KANLinear(config.hidden_dim, 3*config.hidden_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.post_attn_dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.position_encoding = RoPE(config)

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = x.size()

        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.num_attention_heads, -1)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, -1)
        q, k = self.position_encoding(q, k)
        qkt = torch.einsum("bhld,bhmd->bhlm", q, k)
        qkt = qkt / math.sqrt(q.size(-1))
        mask = attention_mask.unsqueeze(1).unsqueeze(1).to(x.device)
        masked_qkt = qkt.masked_fill(~mask, -torch.inf)
        masked_qkt = F.softmax(masked_qkt, dim=-1)
        masked_qkt = self.attn_dropout(masked_qkt)
        output = torch.einsum("bhlm,bhmd->bhld", masked_qkt, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.post_attn_dropout(output)

        return self.out(output)
