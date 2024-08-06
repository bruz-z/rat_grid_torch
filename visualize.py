# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import cv2

from model import Model
from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells

def load_checkpoints(model, checkpoint_dir="./saved/checkpoints"):
    checkpoint = torch.load(checkpoint_dir + '/checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)
  
    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size-1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret

def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret

def convert_to_colomap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im

np.random.seed(1)
torch.manual_seed(1)

data_manager = DataManager()

place_cells = PlaceCells()
hd_cells = HDCells()

data_manager.prepare(place_cells, hd_cells)

model = Model(place_cell_size=place_cells.cell_size,
              hd_cell_size=hd_cells.cell_size,
              sequence_length=100)
model.eval()

# Load checkpoints
# load_checkpoints(model)

batch_size = 10
sequence_length = 100
resolution = 20
maze_extents = 4.3

activations = np.zeros([512, resolution, resolution], dtype=np.float32) # (512, 32, 32)
counts = np.zeros([resolution, resolution], dtype=np.int32) # (32, 32)

index_size = data_manager.get_confirm_index_size(batch_size, sequence_length)

for index in range(index_size):
    out = data_manager.get_confirm_batch(batch_size, sequence_length, index)
    inputs_batch, place_init_batch, hd_init_batch, place_pos_batch = out
    
    inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32)
    place_init_batch = torch.tensor(place_init_batch, dtype=torch.float32)
    hd_init_batch = torch.tensor(hd_init_batch, dtype=torch.float32)
    
    place_pos_batch = np.reshape(place_pos_batch, [-1, 2])
    # (1000, 2)
    
    with torch.no_grad():
        g, _, _ = model(inputs_batch, place_init_batch, hd_init_batch, keep_prob=1.0)
    
    g = g.numpy()

    for i in range(batch_size * sequence_length):
        pos_x = place_pos_batch[i, 0]
        pos_z = place_pos_batch[i, 1]
        x = (pos_x + maze_extents) / (maze_extents * 2) * resolution
        z = (pos_z + maze_extents) / (maze_extents * 2) * resolution
        counts[int(x), int(z)] += 1
        activations[:, int(x), int(z)] += np.abs(g[i, :])

for x in range(resolution):
    for y in range(resolution):
        if counts[x, y] > 0:
            activations[:, x, y] /= counts[x, y]

hidden_size = 512

cmap = matplotlib.cm.get_cmap('jet')

images = []

for i in range(hidden_size):
    im = activations[i, :, :]
    im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im = convert_to_colomap(im, cmap)

    im = cv2.resize(im,
                    dsize=(resolution*2, resolution*2),
                    interpolation=cv2.INTER_NEAREST)
    # (40, 40, 4), uint8
    images.append(im)

concated_image = concat_images_in_rows(images, 32, resolution*2)
cv2.imwrite("grid.png", concated_image)
