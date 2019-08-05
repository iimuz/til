# overlap dif image

指定した基準画像と、それ以外の画像との差分を基準画像に重畳するスクリプト。

## Usage

下記パラメータを python スクリプトに記載する。

- `BASE_IMAGE_NAME`: ベース画像へのパス
- `DIFF_IMAGE_LIST`: 差分画像を取得したい画像パスリスト
- `OUTPUT_IMAGE_NAME`: 出力する画像パス

下記コマンドを実行する。

```sh
$ python overlap_diff_image/overlap.py
```
