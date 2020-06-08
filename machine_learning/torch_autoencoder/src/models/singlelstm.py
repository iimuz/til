import torch.nn as nn
import torch.nn.functional as F


class SingleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device) -> None:
        super(SingleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.output_size, batch_first=True)

    def forward(self, x):
        v = x.view(x.shape[0], -1, x.shape[1])

        code, _ = self.lstm(v)
        code = F.relu(code)

        decode, _ = self.lstm2(code)

        outputs = decode.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs
