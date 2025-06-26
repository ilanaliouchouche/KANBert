from dataclasses import dataclass


@dataclass
class KANBertConfig:

    vocabulary_size: int
    hidden_dim: int
    max_sequence_len: int
    n_layers: int
    intermediate_dim: int
    num_attention_heads: int
    dropout: float = 0.1
    num_experts: int = 3
    top_k_experts: int = 2
    periodicity: int = 10000
