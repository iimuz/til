from torch import nn


class Generator(nn.Module):
    """ Generator
    """

    def __init__(self, z_dim: int, width: int, height: int, channel: int) -> None:
        """ Initialize

        Note:
        ---
        62 次元の入力ベクトルを 1024 次元に拡張し、
        7x7 サイズの 128 チャネル画像へ変換する。
        """
        super().__init__()

        self._WIDTH = width / 4
        self._HEIGHT = height / 4
        self._INTERMIDIATE_CHANNEL = 128

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self._INTERMIDIATE_CHANNEL * self._WIDTH * self._HEIGHT),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                self._INTERMIDIATE_CHANNEL, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """ forward

        Note:
        ---
        input から fc によって 128x7x7 次元のベクトルを生成する。
        deconv には画像を入力としたいため、
        view によって (128, 7, 7) 画像へ変換する。
        """
        x = self.fc(input)
        x = x.view(-1, self._INTERMIDIATE_CHANNEL, self._WIDTH, self._HEIGHT)
        x = self.deconv(x)
        return x
