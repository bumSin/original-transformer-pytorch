# Transformer

Embeddings and Softmax
"Transformer uses learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.
It also uses learned linear transformation and softmax function to convert decoder output to predict next-token probabilities.

It shares same weight matrix between the two embedding layers and the pre softmax linear transformation. In the embedding layers, we multiply those weights by sqrt(d_model)"

Explanation:
1. The embedding tensor before the encoder stack, the embedding tensor before the decoder stack, and the weight tensor of the linear layer that converts the final decoder vector into logits are the same tensor as all three tensors need to be of dimensions d_model x d_vocab and effectively perform the same function.
2. hen you extract a subword's embedding from the embedding tensor before the encoder or decoder stacks, just multiply it with d_model ^ 0.5 before adding it to the positional encoding.


Positional Embeddings

Position embeddings are defined in paper as follows:


<img width="363" alt="image" src="https://github.com/user-attachments/assets/23d94f72-b641-43a9-a56e-ec205f3e9599">

You can find the code to implement this and visualise in the file visualising_position_embeddings.py

Visualising Positional encodings
![image](https://github.com/user-attachments/assets/49d3ac1a-760d-486d-bb71-09dc3507f720)
