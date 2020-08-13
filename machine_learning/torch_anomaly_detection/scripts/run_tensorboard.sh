#!/bin/sh
#
# tensorboard を起動します。

DATA_DIR=${DATA_DIR:-data}
PORT=${TENSORBOARD_PORT:-6006}

tensorboard --logdir $DATA_DIR --host localhost --port $TENSORBOARD_PORT
