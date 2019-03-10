from torch import nn


class Generator(nn.Module):
    """ Generator
    """

    def __init__(self):
        """ Initialize

        Note:
        ---
        62 次元の入力ベクトルを 1024 次元に拡張し、
        7x7 サイズの 128 チャネル画像へ変換する。
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(62, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
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
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x
