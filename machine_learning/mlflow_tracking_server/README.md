# MLFlow Tracking Server Sample

[MLFlow][mlflow] Tracking Server を起動するサンプルコードです。
Artifact ファイルの保存先はローカルであり、 DB は sqlite としています。
下記コマンドでサーバを起動できます。

```sh
poetry install
poetry run bash run_mlflow_tracking_server.sh
```

その他のコマンドとして下記のスクリプトを用意しています。

- `mlflow_gc.sh`:
  UI から実験を削除した場合に Artifact が削除されないため、
  削除した実験の Artifact を削除するコマンドを実行します。

[mlflow]: https://mlflow.org/
