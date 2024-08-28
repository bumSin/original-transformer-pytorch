import torch
from torch import nn

from utils.selfAttentionImpl import SelfAttentionLayer


class Decoder(nn.Module):
    def __init__(self, d_model = 512, d_k = 64, d_v = 64, h = 8, ):
        super().__init__()
        self.masked_attention_layer = SelfAttentionLayer(d_model, d_k, d_v, h, mask=True)  # Add mask as a flag
        self.encoder_decoder_attention_layer = SelfAttentionLayer(d_model, d_k, d_v, h)  # Add input from encoder as one more param
        self.feedForward_layer = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out):
        masked_attention_in = x
        masked_attention_out = self.masked_attention_layer(masked_attention_in)
        residual_masked_attention_out = self.layer_norm(masked_attention_out + masked_attention_in)

        encoder_decoder_attention_in = residual_masked_attention_out
        encoder_decoder_attention_out = self.encoder_decoder_attention_layer(encoder_decoder_attention_in, encoder_out)  # encoder_out is connection from encoders
        residual_out = self.layer_norm(encoder_decoder_attention_out + encoder_decoder_attention_in)

        ff_in = residual_out
        ff_out = self.feedForward_layer(ff_in)
        residual_ff_out = self.layer_norm(ff_out + ff_in)

        return residual_ff_out

# TEST CODE
decoder = Decoder()
x = torch.randn(120, 512)
encoder_out = torch.randn(120, 512)

y = decoder(x, encoder_out)
print(f"y shape: {y.shape}")
