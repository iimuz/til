# third party
import tensorflow as tf
from tensorflow.keras import layers

# my packages
import history


class Autoencoder(tf.keras.Model):
    """単純な全結合Autoencoder
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize

        Args:
            input_dim (int): 入力次元数
        """
        super(Autoencoder, self).__init__()
        dims = 32
        self.encoder = _make_encoder(input_dim, dims)
        self.decoder = _make_decoder(dims, input_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        code = self.encoder(inputs)
        reconstruct = self.decoder(code)
        return reconstruct

    @tf.function
    def train_step(
        self,
        inputs: tf.Tensor,
        loss_obj: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        batch_history: history.Batch = None,
    ) -> None:
        """1バッチに対する学習

        Args:
            inputs (tf.Tensor): バッチ
            loss_obj (tf.keras.losses.Loss): 損失計算オブジェクト
            optimizer (tf.keras.optimizers.Optimizer): 最適化関数
            batch_history (history.Batch, optional): 計算履歴出力用. Defaults to None.
        """
        with tf.GradientTape() as tape:
            reconstruct = self(inputs)
            loss = loss_obj(inputs, reconstruct)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if batch_history is not None:
            batch_history.loss(loss)


def reconstruct(model: Autoencoder, inputs: tf.Tensor) -> tf.Tensor:
    """入力データをモデルを利用して再構成する

    Args:
        model (Autoencoder): 利用するモデル
        inputs (tf.Tensor): 入力データ

    Returns:
        tf.Tensor: 再構成結果
    """
    return model(inputs)


def _make_decoder(input_dim: int, output_dim: int) -> tf.keras.Model:
    """decoderを生成する

    Args:
        input_dim (int): 入力次元
        output_dim (int): 出力次元

    Returns:
        tf.keras.Model: decoderモデル
    """
    model = tf.keras.Sequential([layers.Dense(output_dim, activation="relu")])

    return model


def _make_encoder(input_dim: int, output_dim: int) -> tf.keras.Model:
    """encoderを生成する

    Args:
        input_dim (int): 入力次元
        output_dim (int): 出力次元

    Returns:
        tf.keras.Model: decoderモデル
    """
    model = tf.keras.Sequential([layers.Dense(output_dim, activation="relu")])

    return model
