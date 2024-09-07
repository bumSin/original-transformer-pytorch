import torch
from torch import nn

from model.decoder import DecoderStack
from model.encoder import EncoderStack
from utils.positionalEmbedding import PositionalEmbedding


class Transformer(nn.Module):
    def __init__(self, d_model, h, d_ff, src_vocab_size, tgt_vocab_size, dropout, num_layers = 6):
        super().__init__()

        self.src_embeddings = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embeddings = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_embeddings = PositionalEmbedding(d_model)

        self.Encoders = EncoderStack(num_layers, d_model, h, d_ff, dropout)
        self.Decoders = DecoderStack(num_layers, d_model, h, d_ff, dropout)

        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        # Using log softmax because nn.KLDivLoss() requires log probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)            # Some people club these two layers and call it a generator


    def forward(self, x_src, x_tgt, src_mask=None, tgt_mask=None):  # x.shape (Batch x seq_len)
        '''
        As input we expect batches of sentences; A sentence will be represented by a 1 D vector containing position indices
        of respective word in the vocab.
        :param x:
        :return:
        '''

        # Convert source and target sentence (vector of word indices in vocab vector) into embeddings
        # src = tgt while training same language tasks
        src_embedded = self.src_embeddings(x_src)                   # expected dim: batch_size, seq_len, d_model
        tgt_embedded = self.tgt_embeddings(x_tgt)

        src_embedded = self.pos_embeddings(src_embedded)             # expected dim: batch_size, seq_len, d_model
        tgt_embedded = self.pos_embeddings(tgt_embedded)

        encoder_output = self.Encoders(src_embedded)
        decoder_output = self.Decoders(tgt_embedded, encoder_output, src_mask, tgt_mask)    # Expected dim: batch_size, seq_len, d_model

        tgt_vocab_size_logits = self.final_linear(decoder_output)     # Expected dim: batch_size, seq_len, tgt_vocab_size
        # Converting logits to probabilities
        prob_distribution = self.log_softmax(tgt_vocab_size_logits)

        return prob_distribution

if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1100
    d_model = 512
    num_layers = 6
    h = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 10
    seq_len = 5

    # As input we have a batch of sentences
    x_src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    x_tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

    transformer = Transformer(d_model, h, d_ff, src_vocab_size, tgt_vocab_size, dropout, num_layers)

    output = transformer(x_src, x_tgt, src_mask=None, tgt_mask=None)
    print(output.shape)  # Expected: Batch_size * seq_len * tgt_vocab_size