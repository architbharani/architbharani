This is the Electricity Data Final Project. Here are the steps to run the code:
Load LSTM.pth,LSTM.py in the environment
LSTM.pth is the pretrained model,to use directly,You can see the weights and biases by
# Load the saved model state dictionary
import torch
state_dict = torch.load('LSTM.pth')

import torch
import torch.nn as nn

# Define your LSTM model architecture
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last output from the LSTM
        return out

model = LSTMModel(input_dim=11, hidden_dim=64, num_layers=2, output_dim=11)


# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Access and print the weights and biases
for name, param in model.named_parameters():
    print(f"{name}: {param}")


After Loading LSTM.pth and LSTM.py both, run Model.py

