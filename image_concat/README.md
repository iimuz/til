# image concat

複数画像を並べて 1 枚の画像として保存するスクリプト。

## Usage

下記コマンドを実行する。

```sh
$ cd image_concat
$ python concat.py config.yaml
```

設定値は下記の通り。

- `image`
  - `dir`: 画像が格納されたフォルダパス
  - `query`: 画像ファイルパス取得のためのクエリ
  - `size`:
    - `width`: 結合時に変換する画像サイズ
    - `height`: 結合時に変換する画像サイズ
- `matrix`
  - `columns`: 結合時に横に並べる枚数
- `output`: 出力ファイルパス

## Tips

- 縦方向は、横方向の枚数から自動で算出する。
