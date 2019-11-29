from torch import nn


class Discriminator(nn.Module):
    """ Discriminator
    """

    def __init__(self, width: int, height: int, channel: int) -> None:
        super().__init__()

        self._WIDTH = int(width / 4)
        self._HEIGHT = int(height / 4)
        self._INTERMIDIATE_CHANNEL = 128

        self.conv = nn.Sequential(
            nn.Conv2d(int(channel), 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                64, self._INTERMIDIATE_CHANNEL, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(self._INTERMIDIATE_CHANNEL),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._INTERMIDIATE_CHANNEL * self._WIDTH * self._HEIGHT, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, self._INTERMIDIATE_CHANNEL * self._WIDTH * self._HEIGHT)
        x = self.fc(x)
        return x
