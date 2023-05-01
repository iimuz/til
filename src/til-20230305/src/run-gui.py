"""WebUIを起動する."""
import logging
import sys
from argparse import ArgumentParser
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import gradio as gr
from pydantic import BaseModel

from select_items import AvailableDeviceName, AvailableLanguage, AvailableModelName
from translator import Translator

_logger = logging.getLogger(__name__)


class _RunConfig(BaseModel):
    # スクリプト実行のためのオプション.

    data_dir: Path  # モデルファイルやログファイルなどの記録場所
    verbose: int  # ログレベル


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

    translator = Translator(cache_dir=external_dir)
    with gr.Blocks() as demo:
        source_language = gr.Dropdown(
            choices=[v.value for v in AvailableLanguage],
            label="Source Language",
        )
        target_language = gr.Dropdown(
            choices=[v.value for v in AvailableLanguage],
            label="Source Language",
        )
        device_name = gr.Dropdown(
            choices=[v.value for v in AvailableDeviceName],
            label="Device Name",
        )
        load_model_button = gr.Button("Load model")
        load_model_button.click(
            fn=lambda lang, device: translator.load(
                model_name=AvailableModelName.NLLB_200_DISTILLED_600M,
                source_language=AvailableLanguage(lang),
                device_name=AvailableDeviceName(device),
            ),
            inputs=[
                source_language,
                device_name,
            ],
        )

        source_text = gr.Textbox(label="Source")
        target_text = gr.Textbox(label="Target")
        translate_button = gr.Button("Translate")
        translate_button.click(
            fn=lambda source, lang: translator.translate(
                source=source, target_language=AvailableLanguage(lang), max_length=200
            ),
            inputs=[source_text, target_language],
            outputs=target_text,
        )

    demo.launch()


def _parse_args() -> _RunConfig:
    # スクリプト実行のための引数を読み込む.
    parser = ArgumentParser(description="WebUI for translation using NLLB200.")

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
