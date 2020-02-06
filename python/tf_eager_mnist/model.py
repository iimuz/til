import tensorflow as tf

from logging import getLogger

logger = getLogger(__name__)


class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x


class Trainer:
    def __init__(self):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

    @tf.function
    def train_step(self, model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss_val = self.loss(labels, predictions)
        gradients = tape.gradient(loss_val, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss(loss_val)
        self.train_accuracy(labels, predictions)


class Predictor:
    def __init__(self):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.predict_loss = tf.keras.metrics.Mean(name="predict_loss")
        self.predict_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="predict_accuracy"
        )

    @tf.function
    def predict_step(self, model, images, labels):
        predictions = model(images)
        loss_val = self.loss(labels, predictions)
        self.predict_loss(loss_val)
        self.predict_accuracy(labels, predictions)


class Checkpoint:
    def __init__(self, network, optimizer=None):
        args = {"net": network}
        if optimizer is not None:
            args["optimizer"] = optimizer
        self.ckpt = tf.train.Checkpoint(**args)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, "_data/ckpts", max_to_keep=3
        )
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            logger.info(f"Restored from {self.manager.latest_checkpoint}")
        else:
            logger.info("Initializing from scratch.")

    def save_counter(self):
        return self.ckpt.save_counter.numpy()

    def save(self):
        return self.manager.save()
