"""epubからテキストを抽出するスクリプト."""
import logging
import sys
from argparse import ArgumentParser
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import ebooklib
from ebooklib import epub
from markdownify import markdownify as md
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class _RunConfig(BaseModel):
    # スクリプト実行のためのオプション.

    input: Path  # テキストを抽出するPDFのファイルパス
    output: Path | None  # 抽出したテキストを保存するファイルパス

    data_dir: Path  # ログファイルなどの記録場所
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

    # 入力ファイルが存在することを確認
    if not config.input.exists():
        raise ValueError(f"not exists: {config.input}")

    # 出力先が指定されていない場合は入力ファイルの拡張子変更版を利用する
    output_filepath = config.output
    if output_filepath is None:
        output_filepath = config.input.with_suffix(".md")

    # テキストの抽出しmarkdown形式で保存
    with output_filepath.open("wt", encoding="utf-8") as f:
        book = epub.read_epub(str(config.input))
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_contents = item.get_body_content().decode()
            md_contents = md(html_contents, strip=["a", "img", "table"])
            f.write(md_contents)


def _parse_args() -> _RunConfig:
    # スクリプト実行のための引数を読み込む.
    parser = ArgumentParser(description="Translation using NLLB200.")

    parser.add_argument(
        "input",
        help="File path of the PDF from which the text is to be extracted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="File path to save the extracted text.",
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
