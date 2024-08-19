import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from hd_cells import HDCells
from model import Model
from data_manager import DataManager
from place_cells import PlaceCells
from options import get_options

def load_checkpoints(model, optimizer, checkpoint_path='./saved/checkpoints/checkpoint.pth'):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from '{checkpoint_path}'")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")

# Set seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Initialize data manager and place cells
data_manager = DataManager()
place_cells = PlaceCells()
hd_cells = HDCells()
data_manager.prepare()
args = get_options()
# Initialize model
model = Model(place_cell_size=place_cells.cell_size, hd_cell_size=hd_cells.cell_size)
model.to(args.device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters())

# Optionally load checkpoints
# load_checkpoints(model, optimizer)

# Set model to evaluation mode
model.eval()

batch_size = 10
sequence_length = 100
resolution = 20
maze_extents = 4.3

activations = np.zeros([256, resolution, resolution], dtype=np.float32)
counts = np.zeros([resolution, resolution], dtype=np.int32)

index_size = data_manager.get_confirm_index_size(batch_size, sequence_length)

# Disable gradient computation for evaluation
with torch.no_grad():
    for index in range(index_size):
        out = data_manager.get_confirm_batch(batch_size, sequence_length, index)
        inputs_batch, place_init_batch, place_pos_batch = out

        place_pos_batch = np.reshape(place_pos_batch, [-1, 2])  # Flatten to (batch_size * sequence_length, 2)

        # Move tensors to the correct device
        inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32).to(args.device)
        place_init_batch = torch.tensor(place_init_batch, dtype=torch.float32).to(args.device)

        # Forward pass through the model
        g, _ = model(inputs_batch, place_init_batch, keep_prob=0)
        g = g.cpu().numpy()

        # Accumulate activations and counts
        for i in range(batch_size * sequence_length):
            pos_x, pos_z = place_pos_batch[i]
            x = int((pos_x + maze_extents) / (2 * maze_extents) * resolution)
            z = int((pos_z + maze_extents) / (2 * maze_extents) * resolution)
            if 0 <= x < resolution and 0 <= z < resolution:
                counts[x, z] += 1
                activations[:, x, z] += np.abs(g[i, :])

# Normalize activations
for x in range(resolution):
    for y in range(resolution):
        if counts[x, y] > 0:
            activations[:, x, y] /= counts[x, y]

# Rescale activations to [0, 1]
activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations))

# Plotting the activations
hidden_size = activations.shape[0]
plt.figure(figsize=(15, int(30 * hidden_size / 128)))
for i in range(hidden_size):
    plt.subplot(hidden_size // 8, 8, 1 + i)
    plt.title(f'Neuron {i}')
    plt.imshow(activations[i, :, :], interpolation="gaussian", cmap="jet")
    plt.axis('off')
plt.tight_layout()
plt.savefig('result/result256_dropout0.5.png')
plt.close()
