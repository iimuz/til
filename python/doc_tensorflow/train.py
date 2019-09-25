import dataset
import matplotlib.pyplot as plt
import numpy as np
import preprocess

from logging import getLogger
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


logger = getLogger(__name__)

input_shape = (96, 96, 3)
classes = 10
batch_size = 128
feature_out = 1280
alpha = 0.5
lambda_ = 0.1


def main() -> None:
    (x_train_s, x_ref, y_ref), _ = dataset.get_fasion_mnist()
    x_train_s = preprocess.resize(x_train_s)
    x_ref = preprocess.resize(x_ref)

    np.random.seed(0)
    train(x_train_s, x_ref, y_ref, 20)


def original_loss(y_true, y_pred):
    """ 損失関数
    """
    lc = (
        1
        / (classes * batch_size)
        * batch_size ** 2
        * K.sum((y_pred - K.mean(y_pred, axis=0)) ** 2, axis=[1])
        / ((batch_size - 1) ** 2)
    )
    return lc


def train(x_target, x_ref, y_ref, epoch_num):
    """ 学習
    """
    logger.info("bulid model...")
    network = MobileNetV2(
        include_top=True,
        input_shape=input_shape,
        alpha=alpha,
        # depth_multiplier=1,
        weights="imagenet",
    )
    network.layers.pop()
    for layer in network.layers:
        if layer.name == "block_13_expand":
            break
        else:
            layer.trainable = False

    model_t = Model(inputs=network.input, outputs=network.layers[-1].output)
    model_r = Network(inputs=model_t.input, outputs=model_t.output, name="shared_layer")
    prediction = Dense(classes, activation="softmax")(model_t.output)
    model_r = Model(inputs=model_r.input, outputs=prediction)

    optimizer = SGD(lr=5e-5, decay=0.00005)
    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model_t.compile(optimizer=optimizer, loss=original_loss)

    model_t.summary()
    model_r.summary()

    logger.info(f"x_target shape is {x_target.shape}")
    logger.info(f"x_ref shape is {x_ref.shape}")

    logger.info("training...")
    ref_samples = np.arange(x_ref.shape[0])
    loss, loss_c = [], []
    for epoch in range(epoch_num):
        x_r, y_r, lc, ld = [], [], [], []
        np.random.shuffle(x_target)
        np.random.shuffle(ref_samples)
        for i in range(len(x_target)):
            x_r.append(x_ref[ref_samples[i]])
            y_r.append(y_ref[ref_samples[i]])
        x_r = np.array(x_r)
        y_r = np.array(y_r)
        for i in range(len(x_target) // batch_size):
            batch_target = x_target[i * batch_size : i * batch_size + batch_size]
            batch_ref = x_r[i * batch_size : i * batch_size + batch_size]
            batch_y = y_r[i * batch_size : i * batch_size + batch_size]

            lc.append(
                model_t.train_on_batch(
                    batch_target, np.zeros((batch_size, feature_out))
                )
            )
            ld.append(model_r.train_on_batch(batch_ref, batch_y))
        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))

        if (epoch + 1) % 1 == 0:
            logger.info(f"epoch: {epoch}")
            logger.info(f"Descriptive loss: {loss[-1]}")
            logger.info(f"Compact loss: {loss_c[-1]}")
    network.save("_data/model.h5")

    plt.plot(loss, label="Descriptive loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    plt.plot(loss_c, label="Compact loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    main()
