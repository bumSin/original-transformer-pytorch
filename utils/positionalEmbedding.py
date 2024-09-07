import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model = 512):
        super().__init__()

        positions = torch.arange(0, seq_len, dtype=torch.float32)  # we need position embeddings for all these positions
        embeddings = [self.time_embedding(pos, d_model) for pos in positions]  # Embeddings for all positions
        pos_embeddings = torch.stack(embeddings)     # seq_len x d_model

        # This will put pos_embeddings in the state_dict even if they are not trainable params
        self.register_buffer('pos_embeddings', pos_embeddings)

    def forward(self, x):
        return x + self.pos_embeddings[:x.size(1)].unsqueeze(0)

    def time_embedding(self, pos, d_model):
        assert d_model % 2 == 0

        d_model_half = d_model // 2

        sin_arr = torch.arange(0, d_model_half, dtype=torch.float32)
        sin_arr = (2 / d_model) * sin_arr
        sin_arr = 10000 ** sin_arr
        sin_arr = pos / sin_arr
        cos_arr = sin_arr.clone()  # creates a deep copy

        sin_arr = torch.sin(sin_arr)
        cos_arr = torch.cos(cos_arr)

        # :) A little trick to merge alternate indexes of sin and cosine arrays
        merged = torch.stack((sin_arr, cos_arr), dim=1).flatten()

        return merged


if __name__ == "__main__":
    ## Test code

    batch = 21
    seq_len = 10
    d_model = 512
    pos_enc = PositionalEmbedding(seq_len = 100)
    x = torch.randn(batch, seq_len, d_model)
    y = pos_enc(x)
    print(y)