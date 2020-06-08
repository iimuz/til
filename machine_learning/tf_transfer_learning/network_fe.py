import tensorflow as tf

from tensorflow.python.framework.ops import EagerTensor


class MobileNetV2FE(tf.keras.Model):
    """MobileNetV2 を利用した特徴抽出と分類
    """

    def __init__(self, input_shape=(160, 160, 3)) -> None:
        """初期化

        Args:
            input_shape (tuple, optional): 入力次元. Defaults to (160, 160, 3).
        """
        super(MobileNetV2FE, self).__init__()
        self.base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        self.base.trainable = False
        self.global_ave = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs: EagerTensor):
        """順伝搬

        Args:
            inputs (EagerTensor): 入力

        Returns:
            EagerTensor: 順伝搬結果
        """
        x = self.base(inputs)
        x = self.global_ave(x)
        x = self.prediction_layer(x)
        return x


def _main() -> None:
    """簡易動作テスト用スクリプト
    """
    import tensorflow.compat.v1 as tfv1

    tfv1.enable_eager_execution()

    # base model
    model = MobileNetV2FE()
    model.base.summary()

    # additional model
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.build((32, 160, 160, 3))
    model.summary()


if __name__ == "__main__":
    _main()
