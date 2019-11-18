# zip ファイルの圧縮及び解凍

zip ファイルを python で取り扱うためのサンプルスクリプトです。

`data` フォルダ内のファイルを `temp/new.zip` に圧縮します。
その後、 `temp/new.zip` ファイルを `temp/expand` に解凍します。

## Usage

下記コマンドで、圧縮及び解凍のスクリプトが実行できます。

```sh
python -m unittest test_zip.TestZip.test_zip    # 圧縮の場合
python -m unittest test_zip.TestZip.test_unzip  # 解凍の場合
```

テスト実行の順を制御していないので、一括でテストを実行すると解凍のテストが先に起動し、
圧縮ファイルがないためにエラーする可能性があります。

## 参考資料

- 2018.2.4 [Python で ZIP ファイルを圧縮・解凍する zipfile][nkmk]

[nkmk]: https://note.nkmk.me/python-zipfile/
