import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        v = x.view(x.shape[0], -1)

        h_t, c_t = self.lstm(v)
        outputs = self.linear(h_t)

        outputs = outputs.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs
