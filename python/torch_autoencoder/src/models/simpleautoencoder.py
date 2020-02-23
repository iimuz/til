from logging import getLogger

import torch.nn as nn
import torch.nn.functional as F

logger = getLogger(__name__)


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        code = F.relu(self.encoder(x))
        reconstruct = F.relu(self.decoder(code))
        return reconstruct


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger.info(SimpleAutoencoder(10, 3))
