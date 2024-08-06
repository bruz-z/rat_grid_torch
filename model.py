# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import get_options

class Model(nn.Module):
    def __init__(self, place_cell_size, hd_cell_size, sequence_length):
        super(Model, self).__init__()
        
        self.sequence_length = sequence_length
        self.place_cell_size = place_cell_size
        self.hd_cell_size = hd_cell_size
        
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, batch_first=True)
        self.fc_g = nn.Linear(128, 512, bias=False)
        
        self.fc_place_logits = nn.Linear(512, place_cell_size, bias=False)
        self.fc_hd_logits = nn.Linear(512, hd_cell_size, bias=False)

    def forward(self, inputs, place_init, hd_init, keep_prob=0.5):
        # Init cell and hidden states
        l0 = F.linear(place_init, weight=torch.zeros(128, self.place_cell_size)) + \
             F.linear(hd_init, weight=torch.zeros(128, self.hd_cell_size))
        m0 = F.linear(place_init, weight=torch.zeros(128, self.place_cell_size)) + \
             F.linear(hd_init, weight=torch.zeros(128, self.hd_cell_size))
        
        initial_state = (l0.unsqueeze(0), m0.unsqueeze(0))  # (1, 10, 128)

        rnn_output, rnn_state = self.lstm(inputs, initial_state) # [10, 100, 128]
        
        rnn_output = rnn_output.contiguous().view(-1, 128)
        g = self.fc_g(rnn_output)
        g_dropout = F.dropout(g, p=keep_prob)

        place_logits = self.fc_place_logits(g_dropout)
        hd_logits = self.fc_hd_logits(g_dropout)

        place_outputs_result = F.softmax(place_logits, dim=-1)

        return place_logits, hd_logits, place_outputs_result
        # return place_logits

    def compute_loss(self, place_logits, hd_logits, place_outputs, hd_outputs):
        place_outputs_reshaped = place_outputs.contiguous().view(-1, self.place_cell_size)
        hd_outputs_reshaped = hd_outputs.contiguous().view(-1, self.hd_cell_size)

        place_loss = F.cross_entropy(place_logits, torch.argmax(place_outputs_reshaped, dim=1))
        hd_loss = F.cross_entropy(hd_logits, torch.argmax(hd_outputs_reshaped, dim=1))
        
        return place_loss, hd_loss
