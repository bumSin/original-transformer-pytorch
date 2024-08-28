import torch
from torch import nn

from utils.layerImpls import SelfAttentionLayer


class Encoder(nn.Module):
    def __init__(self, d_model = 512, d_k = 64, d_v = 64, h = 8):
        super().__init__()
        self.self_attention_layer = SelfAttentionLayer(d_model, d_k, d_v, h)
        self.feedForward_layer = nn.Sequential(
                                    nn.Linear(d_model, 4*d_model),
                                    nn.ReLU(),
                                    nn.Linear(4*d_model, d_model)
                                )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        self_attention_in = x
        self_attention_out = self.self_attention_layer(self_attention_in)
        residual_self_attention_out = self.layer_norm(self_attention_out + self_attention_in)

        ff_in = residual_self_attention_out

        ff_out = self.feedForward_layer(ff_in)
        residual_ff_out = self.layer_norm(ff_out + ff_in)

        return residual_ff_out

# TEST CODE
# enc = Encoder()
# x = torch.randn(120, 512)
#
# y = enc(x)
# print(f"y shape: {y.shape}")