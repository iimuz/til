# 入力画像パラメータ
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3

# 学習用パラメータ
BATCH_SIZE = 128
EPOCH_NUM = 25

LEARNING_RATE = 2e-4

# Generator 用パラメータ
Z_DIM = 62

# Data 用パラメータ
NOISE_MEAN = 0
NOISE_SIGMA = 1

# log 用パラメータ
LOG_DIR = "logs"
CHECKPOINT_EACH = 5
CHECKPOINT_IMAGES = 64

# リソース用パラメータ
USE_CUDA = True
