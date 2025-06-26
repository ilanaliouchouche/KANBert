from model.utils import Encoder, KANBertConfig
import torch
import torch.nn as nn


class KANBert(nn.Module):
    """
    Implementation of a Transformer Autoencoder with KAN layers.
    We can also find a implementation of the RoPE instead of the
    the positional embedding layer.
    """

    def __init__(self, config: KANBertConfig) -> None:
        """
        Constructor for the KANBert class.

        Args:
            config (KANBertConfig): Configuration object for the KANBert model.
        """

        super().__init__()

        self.embeddings = nn.Embedding(config.vocabulary_size,
                                       config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            Encoder(config)
            for _ in range(config.n_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:

        x = self.embeddings(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x
