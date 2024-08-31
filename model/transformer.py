import torch
from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder
from utils.positionalEmbedding import PositionalEmbedding
from utils.utils import shift_right


class Transformer(nn.Module):
    def __init__(self, d_model, vocab_size, batch_size):
        super().__init__()

        self.embedding = nn.Linear(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(batch_size, d_model)

        self.encoders = nn.Sequential(
            *[Encoder() for _ in range(6)]
        )

        # self.decoders = nn.Sequential(
        #     *[Decoder() for _ in range(6)]
        # )

        self.decoders = Decoder()

        self.projection = nn.Linear(d_model, vocab_size)
        self.projection.weight.data = self.embedding.weight.data.T     # Sharing the weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # x.shape (Batch x vocab_size)
        embedding = self.embedding(x)  # embedding.shape Batch x d_model
        with_pos_embedding = self.pos_embed(embedding)  # with_pos_embedding.shape Batch x d_model

        encoder_out = self.encoders(with_pos_embedding)  # Batch x d_model

        # shift right, add <sos>/</eos> token at start
        shifted_embedding = shift_right(with_pos_embedding)

        decoder_out = self.decoders(shifted_embedding, encoder_out)
        logits = self.projection(decoder_out)
        prob_distribution = self.softmax(logits)

        return prob_distribution

if __name__ == "__main__":
    batch_size = 200
    vocab_size = 1000
    d_model = 512

    sentence = torch.randn(batch_size, vocab_size)
    transformer = Transformer(d_model, vocab_size, batch_size)

    output = transformer(sentence)
    print(output.shape)  # Batch_size * vocab_size