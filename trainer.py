# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer(object):
    def __init__(self, data_manager, model, args):
        self.data_manager = data_manager
        self.model = model
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
            inputs_batch, place_outputs_batch, hd_outputs_batch, place_init_batch, hd_init_batch = \
                self.data_manager.get_train_batch(self.args.batch_size, self.args.sequence_length)
            # # RuntimeError: Input and hidden tensors are not at the same device, found input tensor at cuda:0 and hidden tensor at cpu
            # inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32).to(self.device)
            # RuntimeError: Input and parameter tensors are not at the same device, found input tensor at cpu and parameter tensor at cuda:0
            inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32)
            place_outputs_batch = torch.tensor(place_outputs_batch, dtype=torch.float32)
            hd_outputs_batch = torch.tensor(hd_outputs_batch, dtype=torch.float32)
            place_init_batch = torch.tensor(place_init_batch, dtype=torch.float32)
            hd_init_batch = torch.tensor(hd_init_batch, dtype=torch.float32)

            self.optimizer.zero_grad()
            # foward
            place_logits, hd_logits, _ = self.model(inputs_batch, place_init_batch, hd_init_batch, keep_prob=0.5)

            place_loss, hd_loss = self.model.compute_loss(place_logits, hd_logits, place_outputs_batch, hd_outputs_batch)
            l2_reg_loss = self.l2_reg * sum(torch.norm(param) for param in self.model.parameters() if 'bias' not in param.name)

            total_loss = place_loss + hd_loss + l2_reg_loss

            total_loss.backward()

            # Apply gradient clipping
            nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()

            if step % 10 == 0:
                self.summary_writer.add_scalar("place_loss", place_loss.item(), step)
                self.summary_writer.add_scalar("hd_loss", hd_loss.item(), step)
                self.summary_writer.add_scalar("total_loss", total_loss.item(), step)


