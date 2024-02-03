This is the Electricity Data Final Project. Here are the steps to run the code:
Load LSTM.pth,LSTM.py and '15-17(2).xlsx' in the environment

LSTM.pth is the pretrained model,to use directly,You can see the weights and biases by

```python
import torch

# Load the saved model state dictionary
state_dict = torch.load('LSTM.pth')

import torch.nn as nn
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

# Initialize the model
model = LSTMModel(input_dim=11, hidden_dim=64, num_layers=2, output_dim=11)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Print the names and values of parameters
for name, param in model.named_parameters():
    print(f"{name}: {param}")

```

After Loading LSTM.pth and LSTM.py both, run Model.py

