"""Deep Temporal Clastering のモデル実装."""
# default packages
import logging

# thrid party packages
import sklearn.cluster as clusetr
import torch.nn as nn
import torch.nn.functional as F

# logger
logger = logging.getLogger(__name__)


class Autoencder(nn.Module):
    def __init__(self) -> None:
        super(Autoencder, self).__init__()
        n_filters = 16
        kernel_size = 3
        self.n_units = [16, 8]
        self.encoder_conv1 = nn.Conv1d(1, n_filters, kernel_size, stride=1, padding=1)
        self.encoder_lstm1 = nn.LSTM(
            n_filters, self.n_units[0], batch_first=True, bidirectional=True
        )
        self.encoder_lstm2 = nn.LSTM(
            self.n_units[0], self.n_units[1], batch_first=True, bidirectional=True
        )

        self.decoder_conv1 = nn.ConvTranspose1d(
            self.n_units[1], 1, 2, stride=2, padding=0
        )

    def decode(self, x):
        """decoding.

        Notes:
            Input dims: [batch, sequence, feature, 1]
            Ouptut dims: [batch, sequence, feature, 1]
        """
        # [batch, feature, sequence]
        v = x.view(x.shape[0], x.shape[1], -1)
        v = v.permute(0, 2, 1)
        decode = self.decoder_conv1(v)

        # [batch, sequence, feature, 1]
        decode = decode.permute(0, 2, 1)
        decode = decode.view(decode.shape[0], decode.shape[1], decode.shape[2], 1)

        return decode

    def encode(self, x):
        """encoding.

        Notes:
            Input dims: [batch, sequence, feature, 1]
            Ouptut dims: [batch, sequence, feature, 1]
        """
        # [batch, feature, sequence]
        v = x.view(x.shape[0], x.shape[1], -1)
        v = v.permute(0, 2, 1)
        code = F.leaky_relu(self.encoder_conv1(v))
        code = F.max_pool1d(code, kernel_size=2)

        # [batch, sequence, feature]
        code = code.permute(0, 2, 1)
        code, _ = self.encoder_lstm1(code)
        code = (code[:, :, : self.n_units[0]] + code[:, :, self.n_units[0] :]) / 2.0
        code = F.leaky_relu(code)
        code, _ = self.encoder_lstm2(code)
        code = (code[:, :, : self.n_units[1]] + code[:, :, self.n_units[1] :]) / 2.0
        code = F.leaky_relu(code)

        # [batch, sequence, feature, 1]
        code = code.view(code.shape[0], code.shape[1], code.shape[2], 1)

        return code

    def forward(self, x):
        """フォワード処理.

        Notes:
            Input dims: [batch, sequence, feature, 1]
            Ouptut dims: [batch, sequence, feature, 1]
        """
        code = self.encode(x)
        decode = self.decode(code)
        return decode


class DTClustering(nn.Module):
    def __init__(self) -> None:
        super(DTClustering, self).__init__()
        self.autoencoder = Autoencder()

    def encode(self, x):
        code = self.autoencoder.encode(x)
        return code

    def forward(self, x):
        decode = self.autoencoder(x)
        return decode


def calc_centeroid(x, network: DTClustering, n_clusters: int):
    """クラスタ中心を計算します.

    Notes:
        Input x: [batch, sequence, feature, 1]
        Output: [batch, hidden sequence, hidden feature, 1]
    """
    code = network.encode(x)
    feature = code.view(code.shape[0], -1)  # [batch, sequence * feature]
    feature = feature.detach().cpu().numpy()
    km = clusetr.KMeans(n_clusters=n_clusters, n_init=10)
    km.fit(feature)
    centers = km.cluster_centers_.reshape(n_clusters, code.shape[1], code.shape[2], 1)

    return centers


def _main() -> None:
    """モデルを確認するためのスクリプト."""
    logging.basicConfig(level=logging.INFO)
    logger.info(DTClustering())


if __name__ == "__main__":
    _main()
