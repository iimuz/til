"""パス関連を管理するモジュール."""
import os
from pathlib import Path

# プロジェクトのデータ保存フォルダ
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()

# 中間データ保存パス
INTERIM_DIR = DATA_DIR / "interim"
# 処理済みデータ保存パス
PROCESSED_DIR = DATA_DIR / "processed"
# 生データ保存パス
RAW_DIR = DATA_DIR / "raw"
