from logging import getLogger

import torch.nn as nn
import torch.nn.functional as F

logger = getLogger(__name__)


class SingleDense(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SingleDense, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        v = x.view(x.shape[0], x.shape[1], -1)

        code = F.relu(self.encoder(v))
        reconstruct = F.relu(self.decoder(code))

        outputs = reconstruct.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return outputs


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger.info(SingleDense(10, 3))
