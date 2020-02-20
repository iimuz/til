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
        v = x.view(x.shape[0], x.shape[1], -1)

        _, (h_t, _) = self.lstm(v)
        code = F.relu(h_t)

        decode, _ = self.lstm2(code)

        outputs = decode.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs
