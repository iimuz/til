# default package
import xml.etree.ElementTree as ET
from logging import getLogger
from typing import Dict

# third party
import pandas as pd
import requests

# logger
logger = getLogger(__name__)


def get_value(element: ET.Element, query: str) -> str:
    """elementの要素をfindして結果をtextとして返す。queryがない場合はNoneを返す。

    Args:
        element (ET.Element): 検索するelemnet
        query (str): 検索クエリ

    Returns:
        str: 検索結果
    """
    val = element.find(query)
    return val.text if val is not None else None


def convert_type(element: ET.Element) -> Dict:
    """elementから要素をDictとして返す。

    Args:
        element (ET.Element): 探索するelement

    Returns:
        Dict: データ
    """
    return {
        "title": get_value(element, "title"),
        "link": get_value(element, "link"),
        "author": get_value(element, "author"),
        "pubDate": get_value(element, "pubDate"),
    }


def _main() -> None:
    """動作テスト用の簡易スクリプト
    """
    # ログ設定
    import logging

    logging.basicConfig(level=logging.INFO)

    # パラメータ設定
    url = "http://iss.ndl.go.jp/api/opensearch"
    params_example = {
        "title": "機械学習",
        "mediatype": "1",
        "from": "2019-01-01",
        "cnt": "10",
        "idx": "1",
    }

    # API に問い合わせ
    response = requests.get(url, params_example)

    # XML 形式で取得できるので pandas.DataFrame 形式に変換
    root = ET.fromstring(response.text.encode("utf-8"))
    df = pd.DataFrame([convert_type(item) for item in root.findall(".//item")])

    # 取得結果を表示
    logger.info(f"=== data ===\n{df.head()}")


if __name__ == "__main__":
    _main()
