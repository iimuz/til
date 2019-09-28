import tensorflow as tf

from tensorflow.python.framework.ops import EagerTensor


class MobileNetV2FT(tf.keras.Model):
    """MobileNetV2 を利用した Fine Tuning
    """

    def __init__(self, input_shape=(160, 160, 3)) -> None:
        """初期化

        Args:
            input_shape (tuple, optional): 入力次元. Defaults to (160, 160, 3).
        """
        super(MobileNetV2FT, self).__init__()
        self.base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        self.base.trainable = True
        self.fine_tune_at = 100
        for layer in self.base.layers[: self.fine_tune_at]:
            layer.trainable = False
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
    import logging
    import tensorflow.compat.v1 as tfv1

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    tfv1.enable_eager_execution()

    # base model
    model = MobileNetV2FT()
    model.base.summary()

    # additional model
    base_learning_rate = 0.0001
    model.build((32, 160, 160, 3))
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    logger.info(f"trainable variables = {len(model.trainable_variables)}")


if __name__ == "__main__":
    _main()
