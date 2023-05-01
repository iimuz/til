"""NLLB200を利用した翻訳."""
import logging
import sys
from argparse import ArgumentParser
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from select_items import AvailableDeviceName, AvailableModelName

_logger = logging.getLogger(__name__)


class _RunConfig(BaseModel):
    # スクリプト実行のためのオプション.

    source_language: str  # 入力文章の言語
    target_language: str  # 翻訳したい言語
    max_length: int  # 出力する最大文字数
    model_name: str  # 利用するモデル名

    loop: bool  # trueの場合は、翻訳を繰り返すモード
    device_name: str  # cpu, cuda, mps の選択肢

    data_dir: Path  # モデルファイルやログファイルなどの記録場所
    verbose: int  # ログレベル


def _get_model_path(model_name: AvailableModelName) -> str:
    # 利用可能なモデル一覧からモデルのパスを返す.
    model_paths = {
        AvailableModelName.NLLB_200_DISTILLED_600M: "facebook/nllb-200-distilled-600M",
        AvailableModelName.NLLB_200_DISTILLED_1_3B: "facebook/nllb-200-distilled-1.3B",
        AvailableModelName.NLLB_200_1_3B: "facebook/nllb-200-1.3B",
        AvailableModelName.NLLB_200_3_3B: "facebook/nllb-200-3.3B",
    }

    return model_paths[model_name]


def _get_device(device_name: AvailableDeviceName) -> torch.device:
    # 指定したデバイスが利用できるか判定して、利用できる場合のみデバイス情報を返す.
    if AvailableDeviceName.CPU == device_name:
        return torch.device("cpu")

    if AvailableDeviceName.CUDA == device_name:
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available.")
        return torch.device("cuda:0")

    if AvailableDeviceName.MPS == device_name:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise ValueError(
                    "MPS not available because the current PyTorch install was not"
                    " built with MPS enabled."
                )
            else:
                raise ValueError(
                    "MPS not available because the current MacOS version is not 12.3+"
                    " and/or you do not have an MPS-enabled device on this machine."
                )
        return torch.device("mps")

    raise ValueError(f"Unknown device name: {device_name}")


def _main() -> None:
    # スクリプトのエントリポイント.

    # 実行時引数の読み込み
    config = _parse_args()

    # 保存場所の初期化
    interim_dir = config.data_dir / "interim"
    interim_dir.mkdir(exist_ok=True)
    external_dir = config.data_dir / "external"
    external_dir.mkdir(exist_ok=True)

    # ログ設定
    loglevel = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(config.verbose, logging.DEBUG)
    _setup_logger(filepath=(interim_dir / "log.txt"), loglevel=loglevel)
    _logger.info(config)

    # デバイスの設定
    device_info = _get_device(AvailableDeviceName(config.device_name))

    # モデルの読み込み
    model_name = _get_model_path(AvailableModelName(config.model_name))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=external_dir,
        src_lang=config.source_language,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=external_dir)
    model = model.to(device_info)

    while True:
        print("input article(Stop CTRL+c):")
        try:
            article = input()
        except KeyboardInterrupt:
            if config.loop:
                _logger.info("use ctrl+c. stop loop.")
                break

            raise
        _logger.info("input buffer: %s", article)

        inputs = tokenizer(article, return_tensors="pt")
        translated_tokens = model.generate(
            **inputs.to(device_info),
            forced_bos_token_id=tokenizer.lang_code_to_id[config.target_language],
            max_length=config.max_length,
        )
        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(f"result: {result}")
        _logger.info("result: %s", result)

        if not config.loop:
            break

    _logger.info("success!")


def _parse_args() -> _RunConfig:
    # スクリプト実行のための引数を読み込む.
    parser = ArgumentParser(description="Translation using NLLB200.")

    parser.add_argument(
        "-s",
        "--source-language",
        default="eng_Latn",
        help="The language of the input text.",
    )
    parser.add_argument(
        "-t",
        "--target-language",
        default="jpn_Jpan",
        help="The language of the output text.",
    )
    parser.add_argument(
        "-l",
        "--max-length",
        default=100,
        type=int,
        help="Maximum number of characters in the output string.",
    )
    parser.add_argument(
        "--model-name",
        default=AvailableModelName.NLLB_200_DISTILLED_600M.value,
        choices=[v.value for v in AvailableModelName],
        type=str,
        help="Name of the model to be used.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeatedly enter and perform translations.",
    )
    parser.add_argument(
        "--device-name",
        default=AvailableDeviceName.CPU.value,
        choices=[v.value for v in AvailableDeviceName],
        type=str,
        help="Select the device to be used.",
    )

    parser.add_argument(
        "--data-dir",
        default=(Path(__file__).parents[1] / "data").resolve(),
        help="Root path of where model files nad log files are saved.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Set the log level for detailed messages.",
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _setup_logger(
    filepath: Path | None,  # ログ出力するファイルパス. Noneの場合はファイル出力しない.
    loglevel: int,  # 出力するログレベル
) -> None:
    # ログ出力設定
    # ファイル出力とコンソール出力を行うように設定する。

    # ファイル出力のログレベルは最低でもINFOとする。
    # debug出力の時はdebugレベルまで出力するようにする。
    minimum_loglevel = loglevel if loglevel <= logging.INFO else logging.INFO
    _logger.setLevel(minimum_loglevel)

    # consoleログ
    console_handler = StreamHandler(stream=sys.stdout)
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(
        Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
    )
    _logger.addHandler(console_handler)

    # ファイル出力するログ
    # 基本的に大量に利用することを想定していないので、ログファイルは多くは残さない。
    if filepath is not None:
        file_handler = RotatingFileHandler(
            filepath,
            encoding="utf-8",
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=1,
        )
        file_handler.setLevel(minimum_loglevel)
        file_handler.setFormatter(
            Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
        )
        _logger.addHandler(file_handler)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
        sys.exit(1)
