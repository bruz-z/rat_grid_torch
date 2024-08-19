import argparse
import torch

def get_options():
    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument("--save_dir", type=str, default="saved", help="Checkpoints, log, options save directory")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length")
    parser.add_argument("--steps", type=int, default=300000, help="Training steps")
    parser.add_argument("--save_interval", type=int, default=500, help="Saving interval")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--gradient_clipping", type=float, default=1e-5, help="Gradient clipping")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    return args










