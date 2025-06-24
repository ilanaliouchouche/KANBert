import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import Node2Vec
from torch.utils.data import Dataset, DataLoader
from model import KANBert, KANBertConfig


class WalkDataset(Dataset):
    def __init__(self, walks, mask_token_id):
        self.walks = walks                             # (N_walks, L)
        self.mask_token_id = mask_token_id

    def __len__(self):
        return self.walks.size(0)

    def __getitem__(self, idx):
        return self.walks[idx]                         # L


def mlm_collate(batch):
    batch = torch.stack(batch)                        # B, L
    input_ids = batch.clone()                         # B, L
    labels = batch.clone()                            # B, L
    mask = torch.rand(input_ids.shape) < 0.15         # B, L
    input_ids[mask] = mask_token_id                   # B, L
    labels[~mask] = -100                              # B, L
    attention_mask = torch.ones_like(input_ids,
                                     dtype=torch.bool)       # B, L
    return input_ids, attention_mask, labels          # B, L & B, L & B, L


dataset = KarateClub()
data = dataset[0]

node2vec = Node2Vec(
    edge_index=data.edge_index,
    embedding_dim=16,
    walk_length=10,
    context_size=5,
    walks_per_node=5,
    p=1.0,
    q=1.0,
    num_negative_samples=0,
    sparse=False,
)
walks = node2vec.pos_sample()                         # N_walks, L+1

vocab_size = data.num_nodes
mask_token_id = vocab_size

config = KANBertConfig(
    vocabulary_size=vocab_size + 1,
    hidden_dim=64,
    max_sequence_len=walks.size(1),
    n_layers=2,
    intermediate_dim=128,
    num_attention_heads=4,
    dropout=0.1,
    periodicity=10000
)
model = KANBert(config)

dataset = WalkDataset(walks, mask_token_id)
loader = DataLoader(dataset, batch_size=16, shuffle=True,
                    collate_fn=mlm_collate)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

model.train()
for input_ids, attention_mask, labels in loader:
    out = model(input_ids, attention_mask)            # B, L, D
    logits = out.reshape(-1, out.size(-1))            # B×L, D
    target = labels.reshape(-1)                       # B×L
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    break
