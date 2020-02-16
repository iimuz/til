import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device) -> None:
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.output_size)

    def forward(self, x):
        v = x.view(x.shape[0], -1)

        h_t = torch.zeros(v.size(0), self.hidden_size, dtype=torch.float32).to(
            self.device
        )
        c_t = torch.zeros(v.size(0), self.hidden_size, dtype=torch.float32).to(
            self.device
        )
        h_t, c_t = self.lstm(v, (h_t, c_t))
        code = F.relu(h_t)

        decode = torch.zeros(code.size(0), self.output_size, dtype=torch.float32).to(
            self.device
        )
        c1_t = torch.zeros(code.size(0), self.output_size, dtype=torch.float32).to(
            self.device
        )
        decode, c1_t = self.lstm2(code, (decode, c1_t))

        outputs = decode.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs
