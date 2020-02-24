"""Deep Temporal Clastering のモデル実装."""
# default packages
import logging

# thrid party packages
import torch.functional as F
import torch.nn as nn

# logger
logger = logging.getLogger(__name__)


class DTClustering(nn.Module):
    def __init__(self) -> None:
        super(DTClustering, self).__init__()
        self.encoder_conv1 = nn.Conv1d(1, 16, 3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv1d(16, 8, 3, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(8, 16, 2, stride=2)
        self.decoder_conv1 = nn.ConvTranspose1d(16, 1, 2, stride=2)

    def forward(self, x):
        code = x.view(x.shape[0], x.shape[1], -1)

        code = F.relu(self.encoder_conv1(code))
        code = F.relu(self.encoder_conv2(code))

        reconstruct = F.relu(self.decoder_conv2(code))
        reconstruct = self.decoder_conv1(reconstruct)

        reconstruct = reconstruct.view(x.shape[0], x.shape[1], x.shape[2], -1)
        return reconstruct


def _main() -> None:
    """モデルを確認するためのスクリプト."""
    logging.basicConfig(level=logging.INFO)
    logger.info(DTClustering())


if __name__ == "__main__":
    _main()
