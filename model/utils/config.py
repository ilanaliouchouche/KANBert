from dataclasses import dataclass

@dataclass
class KANBertConfig:
    vocabulary_size: int
    hidden_dim: int
    max_sequence_len: int
    n_layers: int
    intermediate_dim: int
    num_attention_heads: int
    periodicity: int = 10000
