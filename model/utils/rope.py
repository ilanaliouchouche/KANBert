import torch
from typing import Tuple
from model.utils.config import KANBertConfig


class RoPE(torch.nn.Module):

    def __init__(self,
                 config: KANBertConfig) -> None:

        super().__init__()

        self.attention_head_dim = (config.hidden_dim //
                                   config.num_attention_heads)
        self.max_sequence_len = config.max_sequence_len
        self.periodicity = config.periodicity

        self.register_buffer('rotation_matrix',
                             self._generate_rotation_matrix())

    def _generate_rotation_matrix(self) -> torch.Tensor:

        frequencies = 1.0 / (
                      self.periodicity ** (
                        torch.arange(0, self.attention_head_dim, 2) /
                        self.attention_head_dim))
        indexes = torch.arange(self.max_sequence_len)
        angles = torch.outer(indexes, frequencies).float()

        return torch.polar(torch.ones_like(angles), angles)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_attention_heads, seq_len, attention_head_dim = q.size()

        queries = q.reshape(batch_size,
                            num_attention_heads,
                            seq_len,
                            attention_head_dim // 2,
                            2)
        keys = k.reshape(batch_size,
                         num_attention_heads,
                         seq_len,
                         attention_head_dim // 2,
                         2)

        q_complex = torch.view_as_complex(queries)
        k_complex = torch.view_as_complex(keys)

        rotation_matrix = self.rotation_matrix[:seq_len]
        q_rotated = q_complex * rotation_matrix
        k_rotated = k_complex * rotation_matrix

        new_q = torch.view_as_real(q_rotated)
        new_k = torch.view_as_real(k_rotated)

        new_q = new_q.reshape(batch_size,
                              num_attention_heads,
                              seq_len,
                              attention_head_dim)
        new_k = new_k.reshape(batch_size,
                              num_attention_heads,
                              seq_len,
                              attention_head_dim)

        return new_q, new_k
