import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Model
from trainer import Trainer
from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells
from options import get_options

args = get_options()

def load_checkpoints(model):
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return 0

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        print(f"Loaded checkpoint: {checkpoint_path}, step={step}")
        return step + 1
    else:
        print("Could not find old checkpoint")
        return 0

def save_checkpoints(model, optimizer, step):
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, checkpoint_path)
    print("Checkpoint saved")

def train(trainer, start_step):
    optimizer = optim.Adam(trainer.model.parameters(), lr=args.learning_rate)
    for i in range(start_step, args.steps):
        trainer.train(i)
        if i % args.save_interval == args.save_interval - 1:
            save_checkpoints(trainer.model, optimizer, i)

def main():
    np.random.seed(1)
    torch.manual_seed(1)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_manager = DataManager()

    place_cells = PlaceCells()
    hd_cells = HDCells()

    data_manager.prepare(place_cells, hd_cells)
    
    model = Model(place_cell_size=place_cells.cell_size,
                  hd_cell_size=hd_cells.cell_size,
                  sequence_length=args.sequence_length)
    # global device 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(args.device)
    trainer = Trainer(data_manager, model, args)

    # For Tensorboard log
    log_dir = os.path.join(args.save_dir, "log")
    summary_writer = SummaryWriter(log_dir)

    # Load checkpoints
    start_step = load_checkpoints(model)

    # Train
    train(trainer, start_step)

if __name__ == '__main__':
    main()
