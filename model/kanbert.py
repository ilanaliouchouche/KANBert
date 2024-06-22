import torch
import torch.nn as nn


class KANBert(nn.Module):

    def __init__(self, 
                 vocab, 
                 embedding_dim, 
                 position_size,
                 n_layers,
                 intermediate_dim,
                 num_attention_heads) -> None:

        super().__init__()
        
        # self.embeddings = Embeddings(vocab, embedding_dim, position_size)
        
        #self.layers = nn.ModuleList(
        #    [Encoder(embedding_dim, intermediate_dim, num_attention_heads) for _ in range(n_layers)]
        #)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:

        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x