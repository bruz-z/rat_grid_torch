import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer(object):
    def __init__(self, data_manager, model, args):
        self.data_manager = data_manager
        self.model = model.to(args.device)  # Move model to the specified device
        self.args = args
        self.device = args.device
        
        self._prepare_optimizer()
        self._prepare_summary()
      
    def _prepare_optimizer(self):
        # Apply L2 regularization to output linear layers
        self.l2_reg = self.args.l2_reg
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum
        )
        
    def _prepare_summary(self):
        self.summary_writer = SummaryWriter(log_dir=self.args.save_dir + "/log")

    def train(self, num_steps):
        self.model.train()
        for step in tqdm(range(num_steps), desc="Training Progress"):
            inputs_batch, place_outputs_batch, place_init_batch= \
                self.data_manager.get_train_batch(self.args.batch_size, self.args.sequence_length)
            
            # Move all tensors to the same device as the model
            inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32).to(self.device)
            place_outputs_batch = torch.tensor(place_outputs_batch, dtype=torch.float32).to(self.device)
            place_init_batch = torch.tensor(place_init_batch, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            _, place_outputs = self.model(inputs_batch, place_init_batch, keep_prob=0.5)

            # Compute loss
            place_loss = self.model.compute_loss(place_outputs, place_outputs_batch)
            l2_reg_loss = self.l2_reg * sum(torch.norm(param) for param in self.model.parameters() if param.requires_grad)

            total_loss = place_loss + l2_reg_loss

            # Backward pass
            total_loss.backward()

            # Apply gradient clipping
            nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()

            # Log losses
            if step % 10 == 0:
                self.summary_writer.add_scalar("place_loss", place_loss.item(), step)
                self.summary_writer.add_scalar("total_loss", total_loss.item(), step)



