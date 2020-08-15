"""PyTorch 用データセットの共通モジュール."""
# default packages
import enum


class Mode(enum.Enum):
    """データセットの出力モード."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
