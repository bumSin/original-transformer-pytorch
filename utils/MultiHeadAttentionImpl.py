from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttentionImpl(nn.Module):
    def __init__(self, d_model, h, dropout = 0.1):
        super().__init__()

        self.h = h                  # number of heads, here h = 8, following naming convention from paper
        self.d_model = d_model      # Length of the sequence/sentence
        self.d_k = d_model // h     # naming as per the paper, d_model is nothing but embedding dimensions
        self.d_v = self.d_k         # Again, as per the paper, however, it's not necessary, some implementations can have d_k != d_v

        # Linear projections
        self.linear_q = nn.Linear(d_model, self.d_k * h)        # Wq, here d_model = d_k * h
        self.linear_k = nn.Linear(d_model, self.d_k * h)        # Wk
        self.linear_v = nn.Linear(d_model, self.d_v * h)        # Wv
        self.linear_out = nn.Linear(self.d_v * h, d_model)      # Wo

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value are 512 sequence vector which comes from either prev encoder or decoder
        # depending on this module is being used as self attention or cross attention

        batch_size = query.size(0)         # Batch is the first dimension in our convention

        # Linear projections
        Q = self.linear_q(query)           # Q, K and V are batch_size x seq_len x d_model;
        K = self.linear_k(key)
        V = self.linear_v(value)

        # Split into multiple heads
        # Detailed explanation for this seemingly unintuitive block at the end of this file
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)      # Q, K and V are now (batch_size, sequence_length, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        # Compute attention for each head
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v * self.h)
        # TO DO: Add and explanation

        # Final linear layer
        output = self.linear_out(output)
        return output


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute the scaled dot-product attention.
    """
    d_k = query.size(-1)  # Dimension of the key/query vectors

    # Compute the dot products
    # Assuming query and key have dimensions (batch_size, num_heads, sequence_length, d_k)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / d_k ** 0.5
    # Same as --> scores = torch.matmul(query, key.transpose(-2, -1)) / d_k ** 0.5
    # TO DO: compare and verify the outputs and update in the blog

    # Apply the mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Compute the attention weights/ attention pattern
    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Compute the weighted sum of the values
    output = attention_weights @ value

    return output, attention_weights


'''
Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)   

A note on dimension conversions in the above code block
1.	view(batch_size, -1, self.num_heads, self.d_k) -> splits the d_model dimension into self.num_heads heads, where each head has a dimension of self.d_k 
    So, the shape after view becomes -> (batch_size, sequence_length, self.num_heads, self.d_k)
2.	transpose(1, 2)  -> This operation swaps the sequence_length and num_heads dimensions.
    After the transpose, the new shape becomes -> (batch_size, self.num_heads, sequence_length, self.d_k)
    
    By moving self.num_heads to the second dimension, the model can now treat each head independently during the attention calculation. 
    This allows the attention mechanism to be computed for all heads in parallel within a batch, rather than sequentially across heads.
'''

# Test Code
if __name__ == "__main__":
    d_model = 512
    h = 8
    batch_size = 10
    seq_len = 20

    attention_module = MultiHeadAttentionImpl(d_model, h, dropout = 0.1)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    res = attention_module(query, key, value)
    print(res.shape)     # Expected; batch_size x seq_len x d_model