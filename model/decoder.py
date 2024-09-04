import torch
from torch import nn
from utils.MultiHeadAttentionImpl import MultiHeadAttentionImpl


class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionImpl(d_model, h, dropout)
        self.cross_attn = MultiHeadAttentionImpl(d_model, h, dropout)
        self.feedForward_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),           # d_model = 512; d_ff = 2024 in the paper
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)    # Why 3 layerNorms but only one dropout? Because unlike layernorm, dropout doesn't have any learnable params

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):      # Why two masks? I have explained here --> https://medium.com/@shubham.ksingh.cer14
        # Masked Self Attention
        self_attn_out =  self.self_attn(query = x, key = x, value = x, mask = tgt_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # Encoder-Decoder Attantion or Cross Attention
        cross_attn_out = self.cross_attn(query = x, key = encoder_out, value = encoder_out, mask = src_mask)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        # Feed forward
        ff_out = self.feedForward_layer(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return x

class DecoderStack(nn.Module):
    def __init__(self, num_layers, d_model, h, d_ff, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock( d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)

        return x



# TEST CODE
# decoder = DecoderStack(num_layers = 6, d_model = 512, h = 8, d_ff = 2048, dropout=0.1)
# x = torch.randn(120, 10, 512)   # Batch, seq_len, d_model
# encoder_out = torch.randn(120, 10, 512)   # Batch, seq_len, d_model
#
# y = decoder(x, encoder_out)
# print(f"y shape: {y.shape}")
