# overlap dif image

指定した基準画像と、それ以外の画像との差分を基準画像に重畳するスクリプト。

## Usage

下記コマンドを実行する。

```sh
$ cd overlap_diff_image
$ python overlap.py config.yaml
```

設定値は下記の通り。

- `base_image_path`: 基準画像のパス
- `diff_image_list`: 差分画像を生成する画像パスリスト
- `output_image_path`: 出力ファイルパス
