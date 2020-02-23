from logging import getLogger

import torch.nn as nn
import torch.nn.functional as F

logger = getLogger(__name__)


class SimpleDeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SimpleDeepAutoencoder, self).__init__()
        self.encoded_space_dim = hidden_dim // 2
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_nrm = nn.BatchNorm2d(1)
        self.encoder_fc2 = nn.Linear(hidden_dim, self.encoded_space_dim)
        self.decoder_fc2 = nn.Linear(self.encoded_space_dim, hidden_dim)
        self.decoder_fc1 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        code = self.encoder_fc1(x)
        code = self.batch_nrm(code)
        code = F.relu(code)
        code = self.encoder_fc2(code)
        reconstruct = self.decoder_fc2(code)
        reconstruct = self.decoder_fc1(reconstruct)
        reconstruct = F.relu(reconstruct)
        return reconstruct


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger.info(SimpleDeepAutoencoder(10, 6))
