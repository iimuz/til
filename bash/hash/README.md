# ハッシュ値の計算

ハッシュ値計算を行いファイルに記録しておくスクリプトになります。
指定したディレクトリをルートとして、それ以下のフォルダのファイル全てに対してハッシュ値を計算します。

## Usage

実行前にハッシュ値を計算するファイル群が含まれるフォルダのルートディレクトリの指定と、
ハッシュ値の計算結果を出力するファイルを設定してください。
それぞれ、 `ROOT_DIR` と `HASH_FILE` になります。

```sh
bash hash.sh
```

