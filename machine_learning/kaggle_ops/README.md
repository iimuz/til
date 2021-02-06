# KaggleOps Template

MLFlow を利用して KaggleOps 環境を構築するためのひな形です。
このフォルダをコピーして Kaggle を開始することで MLFlow で実験管理ができるようになることを想定しています。
下記記事とリポジトリを参考にしています。

- 2020.10.5 [KaggleOps を考える ~ MLflow + Colaboratory + Kaggle Notebook ~][gmo]
- GitHub [yuooka/kaggleops-tutorial][yuooka]

[gmo]: https://recruit.gmo.jp/engineer/jisedai/blog/kaggleops-mlflow/
[yuooka]: https://github.com/yuooka/kaggleops-tutorial

## Usage

- GCS にバケットを生成する: `bash scripts/gcp.sh gcs`
- GCS のバケットから mlruns をローカルにコピーする: `bash scripts/gcp.sh cp`

### Colab

Colaboratory で利用する場合は、下記の手順で実行します。

1. `notebooks/colab.ipynb` を Colaboratory で起動
1. 起動後に `.env` を `/content/.env` となるようにアップロード
1. `models/config.yml` の内容を実験に併せて修正 (主に実験名とソースコードの位置を設定)
1. すべてのセルを実行
1. GCS へのアクセス許可(URL からアクセスを許可する)

## Tips

- Google colaboratory は python3.6.9
- PyTorch Lightning の MLFlowLogger で実験を開始してから mlflow で記録する必要がある。
  順番を変えると、 MLFlowLogger は内部で実験を生成(既にある実験を利用しない)するため、 mlflow を呼び出した記録が異なる実験となる。
- PyTorch Lightning の MLFlowLogger では Artifact location を設定できなかったので、簡易版の MLFLow Logger を自作している。
  加えて呼び出し順にかかわらず記録できるように create ではなく active run を取得するように修正。
  active run がない場合に生成する。
  DDP などで失敗するのかもしれないが、現時点では考慮していない。
- MLproject において、`python src/models/trainer.py`とするとモジュールパスを通す必要がある。
  しかしながら、Colab 環境において PYTHONPATH の変更などが必要となるため、 `python -m src.models.trainer` としてモジュール呼び出しにしている。
- Colaboratory を利用するときの `.env` ファイル設定で、特に注意すべきものについて記載。
  - `PYTHONPATH`: Colaboratory の場合は意味がない
  - `MLFLOW_TRACKING_URI`:
    学習完了後に関連ファイルを GCS にアップロードするため、ローカルのフォルダを設定する。
  - `MLFLOW_TRACKING_ARTIFACT_LOCATION`:
    アーティファクトは GCS に直接保存するため、GCS の環境を指定する。
    - e.g.) `gs://{GCS_BUCKET_NAME}/artifacts`
- `.env` の設定での注意
  - `MLFLOW` 関連の変数はローカルファイルを設定する場合でも `file:` はつけない。
    内部での処理で `file:` がない前提で書いている。
