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

## 環境設定

起動する URL や Port 番号、ファイル保存先は .env からの読み込みを想定しています。
.sample.env を参考に.env を作成しスクリプト実行前に読み込んでください。

```sh
env $(cat .env | xargs) |  poetry run bash run_mlflow_tracking_server.sh
```

### Docker

docker を利用して MLFlow Tracking Server を起動することが出来ます。

- Dockerfile: MLFlow Tracking Server を起動するための Docker イメージ作成用
- `docker_command.sh`: docker のイメージ作成からコンテナ起動までのオプションを設定したスクリプト

例えば下記のコマンドでコンテナを起動できます。

```sh
# build image
bash docker_command.sh build

# run contaienr
bash docker_comamnd.sh run
```

## Tips

### Backend に SQLite を設定すると学習中に通信エラーが発生する

環境によるのかもしれませんが、バックエンドに SQLite を設定しているとログの頻度か Artifact のファイルサイズの影響により、通信エラーが発生する場合がありました。
ローカルファイルベースに全て以降すると比較的安定することを確認しています。
(MLFlow Tracking Server を動かす環境にも、それなりのスペックが必要?)
