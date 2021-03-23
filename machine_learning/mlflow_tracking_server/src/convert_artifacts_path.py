"""ファイルベース管理の場合にmlrunsにあるartifactsの設定先を書き換える."""
# default packages
import logging
import os
import pathlib
import sys

# third party packages
import tqdm.autonotebook as tqdm
import yaml

# logger
_logger = logging.getLogger(__name__)


def main() -> None:
    """mlrunsをファイルベースで管理している場合に、meta.yamlのartifact_locationを指定したartifact_storeに変更する."""
    dir_mlruns = pathlib.Path(os.environ.get("LOCAL_MLRUNS", "./mlruns")).resolve()
    old_artifact_store = os.environ.get("OLD_ARTIFACT_STORE", "./artifacts")
    new_artifact_store = os.environ.get(
        "ARTIFACT_STORE", str(pathlib.Path("./artifacts").resolve())
    )
    keys_artifact = ["artifact_location", "artifact_uri"]

    if not dir_mlruns.exists():
        _logger.error(f"mlruns direcotry does not exist: {dir_mlruns}")
        return

    meta_list = list(dir_mlruns.glob("**/meta.yaml"))
    with tqdm.tqdm(meta_list) as pbar:
        for path_meta in pbar:
            pbar.set_description(f"filepath: {path_meta}")

            with open(path_meta) as f:
                meta = yaml.safe_load(f)

            for key in keys_artifact:
                if key not in meta.keys():
                    continue
                meta[key] = meta[key].replace(old_artifact_store, new_artifact_store)
            with open(path_meta, "w") as f:
                f.write(yaml.dump(meta))


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        main()
    except Exception as e:
        _logger.exception(e)
        sys.exit("Failed")
