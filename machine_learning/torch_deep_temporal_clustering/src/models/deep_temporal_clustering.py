"""Deep Temporal Clastering のモデル実装."""
# default packages
import logging

# thrid party packages
import numpy as np
import sklearn.cluster as cluster
import torch
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


class Clustering(nn.Module):
    def __init__(self, n_clusters: int, seq_len: int, feature_num: int) -> None:
        super(Clustering, self).__init__()
        self.alpha = 2.0
        self.n_clusters = n_clusters
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.centroids = nn.Parameter(
            torch.rand(1, n_clusters, seq_len, feature_num), True
        )

    def forward(self, x):
        """Forward.

        Notes:
            x: [batch, sequence, feature, 1]
        """
        v = x.view(x.shape[0], 1, x.shape[1], -1)  # [batch, 1, seq, feat]
        diff = v - self.centroids
        distance = torch.sqrt(torch.sum(diff ** 2, dim=3))
        distance = torch.sum(distance, dim=2)

        q = 1.0 / (1.0 + distance ** 2 / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.permute(1, 0) / torch.sum(q, dim=1)).permute(1, 0)

        return q

    def set_centroids(self, centroids: torch.Tensor) -> None:
        """
        Notes:
            centroids: [n_clusters, timestep, feature_num, 1]
        """
        shape = centroids.shape
        with torch.no_grad():
            self.centroids = nn.Parameter(
                centroids.view(1, shape[0], shape[1], shape[2]), True
            )


class DTClustering(nn.Module):
    def __init__(self) -> None:
        super(DTClustering, self).__init__()
        self.autoencoder = Autoencder()
        self.clustering = Clustering(10, 64, 8)

    def encode(self, x):
        code = self.autoencoder.encode(x)
        return code

    def forward(self, x):
        code = self.autoencoder.encode(x)
        decode = self.autoencoder.decode(code)
        cluster = self.clustering(code)
        return decode, cluster

    def init_centroid(self, x, n_clusters: int) -> None:
        centroid = calc_centeroid(x, self, n_clusters)
        self.clustering.set_centroids(torch.from_numpy(centroid))


def calc_centeroid(x, network: DTClustering, n_clusters: int):
    """クラスタ中心を計算します.

    Notes:
        Input x: [batch, sequence, feature, 1]
        Output: [n_clusters, hidden sequence, hidden feature, 1]
    """
    code = network.encode(x)
    feature = code.view(code.shape[0], -1)  # [batch, sequence * feature]
    feature = feature.detach().cpu().numpy()
    km = cluster.KMeans(n_clusters=n_clusters, n_init=10)
    km.fit(feature)
    centers = km.cluster_centers_.reshape(n_clusters, code.shape[1], code.shape[2], 1)
    centers = centers.astype(np.float32)

    return centers


def target_distribution(q):
    weight = q ** 2 / torch.sum(q, axis=0)
    return (weight.T / torch.sum(weight, axis=1)).T


def _main() -> None:
    """モデルを確認するためのスクリプト."""
    logging.basicConfig(level=logging.INFO)
    logger.info(DTClustering())


if __name__ == "__main__":
    _main()
