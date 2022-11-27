"""MT4の履歴を出力したcsvファイルをDBに取り込むスクリプト."""
import csv
import datetime as dt
import logging
import os
import pprint
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from pydantic import BaseModel

import src.db as db
from src.history import History

_logger = logging.getLogger(__name__)

# csv履歴の時刻に付与するタイムゾーン。デフォルトは日本時間。
_TIMEZONE_HOURS = int(os.environ.get("TIMEZONE_HOURS", "9"))


class _RunConfig(BaseModel):
    # スクリプト実行時のオプション設定.

    drop_tables: bool  # Trueの場合は、データベースにあるテーブルを全て削除
    create_tables: bool  # Trueの場合は、データベースにテーブルを生成

    verbose: int  # ログ出力のレベル


def _parse_arguments() -> _RunConfig:
    # スクリプトの実行時引数の取得.
    parser = ArgumentParser()

    parser.add_argument(
        "--drop-tables",
        action="store_true",
        default=False,
        help="Drop database tables.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        default=False,
        help="Create database tables.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Output log level.",
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _setup_logger(level: int) -> None:
    # ロガーの設定.
    logging.basicConfig(level=level)


def _convert_keys(camel_case_data: dict) -> dict:
    # キーを `lowerCamelCase` から `snake_case` に変更する.
    query = re.compile(r"(?<!^)(?=[A-Z])")
    snake_case_data = {query.sub("_", k).lower(): v for k, v in camel_case_data.items()}

    return snake_case_data


def _to_utc_datetime(datetime_str: str, tz_info: dt.timezone) -> dt.datetime:
    # 時刻の文字列から設定されたタイムゾーンを考慮してUTC時間を返す.
    datetime_naive = dt.datetime.strptime(datetime_str, "%Y.%m.%d %H:%M:%S")
    datetime_aware = datetime_naive.replace(tzinfo=tz_info)
    datetime_utc = datetime_aware.astimezone(dt.timezone.utc)

    return datetime_utc


def _main() -> None:
    # MT4の履歴csvファイルをDBに取り込むスクリプト.
    config = _parse_arguments()  # 実行時引数の処理

    # ログ設定
    log_level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(config.verbose, 0)
    _setup_logger(log_level)
    _logger.info(f"config: {pprint.pformat(config.dict())}")

    # データベースのテーブルを削除
    if config.drop_tables:
        _logger.info("Drop all tables...")
        db.Base.metadata.drop_all(bind=db.ENGINE)

    # データベースの初期化
    # テーブルを削除した場合も初期化のために作成が必要
    if config.create_tables or config.drop_tables:
        _logger.info("Create table...")
        db.Base.metadata.create_all(bind=db.ENGINE)

    # DB設定
    session = db.Session()

    # 既存の履歴一覧を取得
    exist_ticket_list: list[tuple[int]] = session.query(History.ticket).all()
    ticket_exist: set[int] = set([v[0] for v in exist_ticket_list])

    # 対象ファイルの取得
    filelist_generator = Path("data/raw").glob("**/*.csv")

    # 文字列から自国へ変換するための時刻情報を持つ列リスト
    column_names_datetime = ["open_time", "close_time", "expiration"]
    column_names_float = [
        "open_price",
        "lots",
        "stop_loss",
        "take_profit",
        "close_price",
        "commission",
        "swap",
        "profit",
    ]
    column_names_int = ["ticket", "type", "magic_number"]
    timezone_info = dt.timezone(dt.timedelta(hours=_TIMEZONE_HOURS))

    for filepath in filelist_generator:
        _logger.debug(f"target file: {filepath}")
        with filepath.open("rt", newline="") as f:
            reader = csv.DictReader(f)
            records: list[History] = list()
            for lower_camel_case_record in reader:
                snake_case_record = _convert_keys(lower_camel_case_record)
                for column_datetime in column_names_datetime:
                    snake_case_record[column_datetime] = _to_utc_datetime(
                        snake_case_record[column_datetime], tz_info=timezone_info
                    )
                for column_float in column_names_float:
                    snake_case_record[column_float] = float(
                        snake_case_record[column_float]
                    )
                for column_int in column_names_int:
                    snake_case_record[column_int] = int(snake_case_record[column_int])
                item = History(**snake_case_record)
                if item.ticket in ticket_exist:
                    _logger.info(f"already exists. skip. ticket={item.ticket}")
                    continue
                records.append(item)
                ticket_exist.add(item.ticket)
        session.add_all(records)
        session.commit()

    _logger.info("success!")


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
        sys.exit(1)
