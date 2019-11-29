import argparse
import pathlib
import shutil
import zipfile
from typing import Dict
from urllib import request


def get_url_sample() -> [str, str]:
    """ sample 版の URL を取得する
    """
    FILENAME = "LLD-icon_sample.zip"
    TARGET_URL = f"https://data.vision.ee.ethz.ch/sagea/lld/data/{FILENAME}"
    return (FILENAME, TARGET_URL)


def get_url_full() -> [str, str]:
    """ full 版の URL を取得する
    """
    FILENAME = "LLD-icon_full_data_PNG.zip"
    TARGET_URL = f"https://data.vision.ee.ethz.ch/sagea/lld/data/{FILENAME}"
    return (FILENAME, TARGET_URL)


def parse_arguments() -> Dict[str, any]:
    """ 引数から設定を行います
    """
    parser = argparse.ArgumentParser(usage=f"Usage python {__file__}")
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        dest="is_sample",
        help="if set this flag, download sample datasets.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        dest="is_full",
        help="if set this flag, download full datasets.",
    )
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    print(f"parameters: {args}")

    DOWNLOAD_DIR = pathlib.Path("./data")
    EXTRACT_DIR = pathlib.Path("./data/icons")
    if args.is_full:
        FILENAME, TARGET_URL = get_url_full()
    elif args.is_sample:
        FILENAME, TARGET_URL = get_url_sample()
    else:
        print(f"you did not select dataset version.")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH = DOWNLOAD_DIR.joinpath(FILENAME)
    with request.urlopen(TARGET_URL) as res, open(str(DOWNLOAD_PATH), "wb") as f:
        shutil.copyfileobj(res, f)

    if not DOWNLOAD_PATH.exists():
        print(f"download error: {TARGET_URL}")
        return
    print(f"download from {TARGET_URL} to {DOWNLOAD_PATH}")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(DOWNLOAD_PATH), "r") as f:
        f.extractall(str(EXTRACT_DIR))
    print(f"expand data from {DOWNLOAD_PATH} to {EXTRACT_DIR}")


if __name__ == "__main__":
    run()
