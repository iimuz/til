import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device) -> None:
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        v = x.view(x.shape[0], -1)
        outputs = []

        h_t = torch.zeros(v.size(0), self.hidden_size, dtype=torch.float32).to(
            self.device
        )
        c_t = torch.zeros(v.size(0), self.hidden_size, dtype=torch.float32).to(
            self.device
        )

        for i, input_t in enumerate(v.chunk(v.size(1), dim=1)):
            h_t, c_t = self.lstm(v, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs = outputs.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs
