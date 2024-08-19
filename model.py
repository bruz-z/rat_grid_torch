import torch
import torch.nn as nn
import torch.nn.functional as F
from options import get_options

class Model(nn.Module):
    def __init__(self, place_cell_size, hd_cell_size):
        super(Model, self).__init__()
        
        self.place_cell_size = place_cell_size
        self.hd_cell_size = hd_cell_size
        
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, batch_first=True)
        self.fc_g = nn.Linear(128, 256)
        self.fc_place_logits = nn.Linear(256, place_cell_size, bias=False)
        self.fc_hd_logits = nn.Linear(256, hd_cell_size, bias=False)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, inputs, place_init, keep_prob):
        args = get_options()
        device = args.device

        # Ensure place_init is on the correct device
        place_init = place_init.to(device)

        # Initialize cell and hidden states
        l0 = torch.zeros(place_init.size(0), 128, device=device)
        m0 = torch.zeros(place_init.size(0), 128, device=device)
        
        initial_state = (l0.unsqueeze(0), m0.unsqueeze(0))  # (1, batch_size, 128)

        # Forward pass through LSTM
        rnn_output, _ = self.lstm(inputs, initial_state)  # [batch_size, seq_len, 128]
        
        # Reshape output and pass through fully connected layers
        rnn_output = rnn_output.contiguous().view(-1, 128)
        g = self.fc_g(rnn_output)
        g_dropout = F.dropout(g, p=keep_prob, training=self.training)

        # Get final output
        place_outputs = self.fc2(g_dropout)

        return g_dropout, place_outputs

    def compute_loss(self, place_outputs, place_outputs_batch):
        # place_outputs_batch: [batch_size, sequence_length, num_features]
        # place_outputs: [batch_size * sequence_length, num_features]
        # Flatten place_outputs to match the shape of place_outputs_batch
        place_outputs_batch = place_outputs_batch.view(-1, place_outputs_batch.size(-1))

        # Compute Mean Squared Error loss
        mse_loss = F.mse_loss(place_outputs, place_outputs_batch)
        return mse_loss
