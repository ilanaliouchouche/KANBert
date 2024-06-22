from .utils import Encoder, KANBertConfig
import torch
import torch.nn as nn

class KANBert(nn.Module):

    def __init__(self, config: KANBertConfig) -> None:
        super().__init__()
        
        self.embeddings = nn.Embedding(config.vocabulary_size, config.hidden_dim)

        self.layers = nn.ModuleList([
            Encoder(hidden_dim=config.hidden_dim,
                    num_attention_heads=config.num_attention_heads,
                    max_sequence_len=config.max_sequence_len,
                    intermediate_dim=config.intermediate_dim,
                    periodicity=config.periodicity) for _ in range(config.n_layers)
        ])
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x