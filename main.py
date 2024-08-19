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

def load_checkpoints(model, optimizer, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        print(f"Loaded checkpoint: {checkpoint_path}, step={step}")
        return step
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

def save_checkpoints(model, optimizer, step, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step}")

def train(trainer, optimizer, start_step, save_interval, steps, checkpoint_dir):
    for step in range(start_step, steps):
        trainer.train(step)
        if step % save_interval == save_interval - 1:
            save_checkpoints(trainer.model, optimizer, step, checkpoint_dir)

def main():
    args = get_options()
    
    # Set seeds for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)
    
    # Create save and checkpoint directories
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize data, model, and trainer
    data_manager = DataManager()
    place_cells = PlaceCells()
    hd_cells = HDCells()
    data_manager.prepare()

    model = Model(place_cell_size=place_cells.cell_size, hd_cell_size=hd_cells.cell_size)
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(data_manager, model, args)

    # For Tensorboard logging
    log_dir = os.path.join(args.save_dir, "log")
    summary_writer = SummaryWriter(log_dir)

    # Load checkpoints
    start_step = load_checkpoints(model, optimizer, checkpoint_dir)

    # Train model
    train(trainer, optimizer, start_step, args.save_interval, args.steps, checkpoint_dir)

if __name__ == '__main__':
    main()
