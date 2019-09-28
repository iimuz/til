import datasets
import network_ft
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import utils

from logging import getLogger

logger = getLogger(__name__)


def _main() -> None:
    """スクリプトのエントリポイント
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    tfv1.enable_eager_execution()

    raw_train, raw_validation, _, metadata = datasets.get_batch_dataset(shuffle_seed=0)
    base_learning_rate = 0.0001
    model = network_ft.MobileNetV2FT()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # initial model accuracy
    loss0, accuracy0 = model.evaluate(raw_validation, steps=20)
    logger.info(f"initial loss: {loss0:.2f}, acc: {accuracy0:.2f}")

    # training
    checkpoint = utils.load_checkpoints(model, save_dir="_data/ckpt_finetuning")
    history = model.fit(
        raw_train, epochs=2, validation_data=raw_validation, callbacks=[checkpoint]
    )
    utils.plot_history(history)


if __name__ == "__main__":
    _main()
