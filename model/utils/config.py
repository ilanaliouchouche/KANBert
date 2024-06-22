from dataclasses import dataclass


@dataclass
class KANBertConfig:
    """
    Configuration class for KANBert model.
    Args:
        vocabulary_size (int): Size of the vocabulary.
        hidden_dim (int): Dimension of the hidden states.
        max_sequence_len (int): Maximum sequence length.
        n_layers (int): Number of layers in the model.
        intermediate_dim (int): Dimension of the intermediate layer
                                in the position wise feed forward.
        num_attention_heads (int): Number of attention heads.
        periodicity (int): Periodicity of the sinusoidal positional encoding.
    """

    vocabulary_size: int
    hidden_dim: int
    max_sequence_len: int
    n_layers: int
    intermediate_dim: int
    num_attention_heads: int
    periodicity: int = 10000
