import torch
import matplotlib.pyplot as plt
import numpy as np


def time_embedding(pos, d_model):
    assert d_model % 2 == 0

    d_model_half = d_model // 2

    sin_arr = torch.arange(0, d_model_half, dtype=torch.float32)
    sin_arr = (2 / d_model) * sin_arr
    sin_arr = 10000 ** sin_arr
    sin_arr = pos / sin_arr
    cos_arr = sin_arr.clone()  # creates a deep copy

    sin_arr = torch.sin(sin_arr)
    cos_arr = torch.cos(cos_arr)

    merged = torch.stack((sin_arr, cos_arr), dim=1).flatten()

    return merged

# Define parameters
d_model = 200 # Model dimension (must be even)
pos = 500 # Define how many positions you want to visualise
positions = torch.arange(0, pos, dtype=torch.float32)  # Example positions

# Calculate embeddings for different positions
embeddings = [time_embedding(pos, d_model) for pos in positions]

# Convert embeddings to NumPy array
embeddings_np = torch.stack(embeddings).numpy()

# Normalize embeddings for visualization
# Here we assume embeddings are in the range [-1, 1]. Adjust normalization if needed.
embeddings_min = embeddings_np.min()
embeddings_max = embeddings_np.max()
embeddings_normalized = (embeddings_np - embeddings_min) / (embeddings_max - embeddings_min)

# Plot the embeddings as an image
plt.figure(figsize=(10, 6))
plt.imshow(embeddings_normalized, cmap='viridis', aspect='auto')
plt.colorbar(label='Normalized Embedding Value')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Time Embeddings Visualization')
plt.show()
