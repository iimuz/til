from tensorflow.keras import backend as K

classes = 10
batch_size = 128


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
