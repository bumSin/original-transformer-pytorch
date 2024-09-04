import torch
from torch import nn

from utils.MultiHeadAttentionImpl import MultiHeadAttentionImpl


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention_layer = MultiHeadAttentionImpl(d_model, h, dropout)
        self.feedForward_layer = nn.Sequential(
                                    nn.Linear(d_model, d_ff),      # d_ff = 2048 in paper
                                    nn.ReLU(),
                                    nn.Linear(d_ff, d_model)
                                )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)    # Since dropout doesn't have any learnable params, we will use this same layer object at both the places

    def forward(self, x):
        '''
        Expected dimensions of input are (Batch_size x seq_len x d_model)
        From the paper: Page 7, chapter 5.4
        We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        '''
        self_attn_output = self.self_attention_layer(query = x, key = x, value = x, mask=None)
        x = x + self.dropout(self_attn_output)      # residual connection
        x = self.norm1(x)

        # Though, as per many sources on internet it worked better to normalise BEFORE the residual connection
        # So the follwoing might work better
        # x = x + self.norm(self.dropout(self_attn_output))

        ff_output = self.feedForward_layer(x)
        x = x + self.dropout(ff_output)  # residual connection
        x = self.norm2(x)

        return x

class EncoderStack(nn.Module):
    def __init__(self, num_layers, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x



# TEST CODE
# enc = EncoderStack(num_layers = 6, d_model = 512, h = 8, d_ff = 2048, dropout=0.1)
# x = torch.randn(120, 10, 512)   # Batch, seq_len, d_model
#
# y = enc(x)
# print(f"y shape: {y.shape}")