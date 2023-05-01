"""選択肢となりえるデータ群."""
from enum import Enum


class AvailableDeviceName(Enum):
    """利用可能なデバイス名一覧."""

    CPU: str = "cpu"
    CUDA: str = "cuda"
    MPS: str = "mps"


class AvailableLanguage(Enum):
    """入出力で選択可能な言語."""

    JPN: str = "jpn_Jpan"
    ENG_LATN: str = "eng_Latn"


class AvailableModelName(Enum):
    """利用可能なモデル名一覧."""

    NLLB_200_DISTILLED_600M: str = "nllb-200-distiled-600M"
    NLLB_200_DISTILLED_1_3B: str = "nllb-200-distiled-1.3B"
    NLLB_200_1_3B: str = "nllb-200-1.3B"
    NLLB_200_3_3B: str = "nllb-200-3.3B"
