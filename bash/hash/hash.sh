#!/bin/bash
#
# 指定したディレクトリ以下のハッシュ値を再帰的に計算しファイルに保存します。

ROOT_DIR=/path/to/root/dir
HASH_FILE=$(pwd)/data/hash.csv

echo "root dir: $ROOT_DIR"
echo "hash csv: $HASH_FILE"

pushd $ROOT_DIR
find . -type f | xargs -IXXX sha256sum XXX | tee $HASH_FILE
popd

