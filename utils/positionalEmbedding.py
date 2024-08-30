import torch
from torch import nn

from visualising_position_embeddings import time_embedding


class PositionalEmbedding(nn.Module):
    def __init__(self, batch_size = 100, d_model = 512):
        super().__init__()

        positions = torch.arange(0, batch_size, dtype=torch.float32)  # we need position embeddings for all these positions
        embeddings = [time_embedding(pos, d_model) for pos in positions]  # Embeddings for all positions
        self.embeddings_np = torch.stack(embeddings)     # B x d_model

    def forward(self, x):
        return x + self.embeddings_np



if __name__ == "__main__":
    ## Test code
    x = torch.ones(100, 512)
    pos_enc = PositionalEmbedding()

    y = pos_enc(x)
    print(y)