from logging import getLogger

import torch.nn as nn
import torch.nn.functional as F

logger = getLogger(__name__)


class DoubleDense(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(DoubleDense, self).__init__()
        self.encoded_space_dim = hidden_dim // 2
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_nrm = nn.BatchNorm2d(1)
        self.encoder_fc2 = nn.Linear(hidden_dim, self.encoded_space_dim)
        self.decoder_fc2 = nn.Linear(self.encoded_space_dim, hidden_dim)
        self.decoder_fc1 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        v = x.view(x.shape[0], -1, x.shape[1])

        code = self.encoder_fc1(v)
        code = self.batch_nrm(code)
        code = F.relu(code)
        code = self.encoder_fc2(code)
        reconstruct = self.decoder_fc2(code)
        reconstruct = self.decoder_fc1(reconstruct)
        reconstruct = F.relu(reconstruct)

        outputs = reconstruct.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger.info(DoubleDense(10, 6))
