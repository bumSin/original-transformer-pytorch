from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def applymask(scaled_score):
    return scaled_score


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, mask=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.mask=mask

        self.wq = nn.Parameter(torch.randn(h, d_model, d_k))
        self.wk = nn.Parameter(torch.randn(h, d_model, d_k))
        self.wv = nn.Parameter(torch.randn(h, d_model, d_v))
        self.wo = nn.Parameter(torch.randn(d_k * h, d_model))

    def forward(self, x, y=None):        # x and y are (Batch x d_model)
        if y is None:                    # For encoder blocks, x=y. For decoder blocks y is output of final encoder
            y = x

        q = torch.matmul(x, self.wq)      # q is (h, B, d_k)   Every head will have it's own query vector
        k = torch.matmul(y, self.wk)      # k is (h, B, d_k)
        v = torch.matmul(y, self.wv)      # v is (h, B, d_k)

        #calculate attention
        # Use einsum to rearrange dimensions from (h, B, d_k) to (h, d_k, B)
        k = torch.einsum('hbd->hdb', k)

        # Similarity of every word with each other Q dot K.T for all h heads
        score = torch.matmul(q, k)  # h x B x B

        # scale down by sqrt(d_k) i.e 8
        scaled_score = score / sqrt(self.d_k)

        if self.mask:
            scaled_score = applymask(scaled_score)

        # Softmaxing over every row in every head
        softmax = F.softmax(scaled_score, dim=2)  # h x B x B

        # Weighted sum of all values to get z
        z = torch.matmul(softmax, v)  # (h B B) x (h B d_k) = (h B d_k)   batch matrix multiplication

        # Concat outputs of all attention heads next to each other
        matrix_reduced = rearrange(z, 'h b d -> b (h d)')     # B x d_k(h)

        o = torch.matmul(matrix_reduced, self.wo)  # B x d_model

        return o