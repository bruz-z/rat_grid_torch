import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells

def load_checkpoints(model, optimizer, checkpoint_dir='./saved/checkpoints'):
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

np.random.seed(1)
torch.manual_seed(1)

data_manager = DataManager()

place_cells = PlaceCells()
hd_cells = HDCells()

data_manager.prepare(place_cells, hd_cells)

model = Model(place_cell_size=place_cells.cell_size,
              hd_cell_size=hd_cells.cell_size,
              sequence_length=100)

device = torch.device('cpu')
model.to(device)

optimizer = optim.Adam(model.parameters())

# Load checkpoints
# load_checkpoints(model, optimizer)
model.eval()

batch_size = 10
sequence_length = 100 ##lstm模型默认值？？？
resolution = 20
maze_extents = 4.3

activations = np.zeros([512, resolution, resolution], dtype=np.float32)  # (512, 32, 32)
counts = np.zeros([resolution, resolution], dtype=np.int32)  # (32, 32)

index_size = data_manager.get_confirm_index_size(batch_size, sequence_length)

with torch.no_grad():
    for index in range(index_size):
        out = data_manager.get_confirm_batch(batch_size, sequence_length, index)
        inputs_batch, place_init_batch, hd_init_batch, place_pos_batch = out

        place_pos_batch = np.reshape(place_pos_batch, [-1, 2])  # (1000, 2)

        inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32).to(device)
        place_init_batch = torch.tensor(place_init_batch, dtype=torch.float32).to(device)
        hd_init_batch = torch.tensor(hd_init_batch, dtype=torch.float32).to(device)

        g ,_ , _ = model(inputs_batch, place_init_batch, hd_init_batch, keep_prob=0.9)
        g = g.cpu().numpy()
        for i in range(batch_size * sequence_length):
            pos_x = place_pos_batch[i, 0]
            pos_z = place_pos_batch[i, 1]
            x = int((pos_x + maze_extents) / (maze_extents * 2) * resolution)
            z = int((pos_z + maze_extents) / (maze_extents * 2) * resolution)
            if 0 <= x < resolution and 0 <= z < resolution:
                counts[x, z] += 1
                activations[:, x, z] += np.abs(g[i, :])

for x in range(resolution):
    for y in range(resolution):
        if counts[x, y] > 0:
            activations[:, x, y] /= counts[x, y]
activations = (activations - np.min(activations)) / (np.max(activations) - np.min(activations))

hidden_size = 512

plt.figure(figsize=(15, int(30 * hidden_size / 128)))
for i in range(hidden_size):
    plt.subplot(hidden_size // 8, 8, 1 + i)
    plt.title('Neuron ' + str(i))
    plt.imshow(activations[i, :, :], interpolation="gaussian", cmap="jet")
    plt.axis('off')
plt.savefig('result/result512_dropout0.9.png')
