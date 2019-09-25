import dataset
import matplotlib.pyplot as plt
import numpy as np
import preprocess
import tensorflow as tf

from logging import getLogger
from tensorflow.keras.applications import MobileNetV2
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


logger = getLogger(__name__)
input_shape = (96, 96, 3)
alpha = 0.5


def main() -> None:
    (x_train_s, _, _), (x_test_s, x_test_b) = dataset.get_fasion_mnist()
    x_train_s = preprocess.resize(x_train_s)
    x_test_s = preprocess.resize(x_test_s)
    x_test_b = preprocess.resize(x_test_b)

    network = tf.keras.models.load_model("_data/model.h5")
    network.summary()

    predict(x_train_s, x_test_s, x_test_b, network)


def predict(x_train_s, x_test_s, x_test_b, model) -> None:
    train = model.predict(x_train_s)
    test_s = model.predict(x_test_s)
    test_b = model.predict(x_test_b)

    train = train.reshape((len(x_train_s), -1))
    test_s = test_s.reshape((len(x_test_s), -1))
    test_b = test_b.reshape((len(x_test_b), -1))

    ms = MinMaxScaler()
    train = ms.fit_transform(train)
    test_s = ms.transform(test_s)
    test_b = ms.transform(test_b)

    clf = LocalOutlierFactor(n_neighbors=5)
    _ = clf.fit(train)

    z1 = -clf._decision_function(test_s)
    z2 = -clf._decision_function(test_b)

    y_true = np.zeros(len(test_s) + len(test_b))
    y_true[len(test_s) :] = 1  # normal = 0, abnormal = 1

    fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((z1, z2)))
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"DOC(AUC = {auc}")
    plt.legend()
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    main()
