"""epubからテキストを抽出するスクリプト."""
import json
import logging
import sys
from argparse import ArgumentParser
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests
from pydantic import BaseModel
from moviepy.editor import concatenate_audioclips, AudioFileClip

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

    # テキストをwavへ変換
    speaker_id = 2
    with config.input.open("rt", encoding="utf-8") as read_f:
        for text_idx, text in enumerate(read_f.readlines()):
            text = text.rstrip()
            if text == "" or text[0] == "#" or len(text) < 2:
                _logger.debug("skip. text is empty.")
                continue
            _logger.debug(f"text: len={len(text)}: {text}")

            res1 = requests.post(
                "http://127.0.0.1:50021/audio_query",
                params={"text": text, "speaker": speaker_id},
            )
            res2 = requests.post(
                "http://127.0.0.1:50021/synthesis",
                params={"speaker": 1},
                data=json.dumps(res1.json()),
            )
            with open(f"data/processed/output{text_idx:05d}.wav", "wb") as write_f:
                write_f.write(res2.content)

    # wavファイルを結合してmp3に変換
    # 全ファイルを一括で実施すると使用メモリ量が多くなるため500行を目安に区切る。
    # 実際には1行に含まれる文字数に依存するため暫定処置
    wav_filelist = sorted(Path("data/processed/").glob("output*.wav"))
    _logger.debug(f"wave filepath len: {len(wav_filelist)}")
    clips = list()
    for wav_index, filepath in enumerate(wav_filelist):
        clips.append(AudioFileClip(str(filepath)))
        if ((wav_index + 1) % 500) == 0:
            _logger.debug(f"save file index: {wav_index + 1}")
            final_clip = concatenate_audioclips(clips)
            final_clip.write_audiofile(
                str(Path(f"data/processed/final{wav_index + 1:05d}.mp3"))
            )
            clips = list()
    if len(clips) > 0:
        _logger.debug("save rest file.")
        final_clip = concatenate_audioclips(clips)
        final_clip.write_audiofile(
            str(Path(f"data/processed/final{wav_index + 1:05d}.mp3"))
        )


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
