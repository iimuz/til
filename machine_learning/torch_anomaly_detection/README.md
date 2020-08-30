# Anomaly Detection

## MLFlow Tracking Server

実験管理として [MLFlow][mlflow] を利用しています。
共通の MLFlow Tracking Server を用意している場合は、
.env に `MLFLOW_TRACKING_URI` として対象の URI を指定してください。
共通の MLFlow Tracking Server がない場合は、
下記コマンドでローカルに MLFlow Tracking Server を起動できます。

```sh
poetry run mlflow ui
```

[mlflow]: https://mlflow.org/

## 既知の不思議な点

- PyTorch Lightning の TrainResult を利用した場合、
  `on_step` を `True` にすることでステップごとのログが記録できるはずですが、
  エポック単位でしかログが記録できていないです。
  そのため、追加でステップ単位のログを記録する部分を追加で記載しています。

## ToDo

- [ ] Test を行い精度を算出するノートブックを追加する。
- [ ] VAE コードを PyTorch Lighting 0.9.x 系に合わせる。
- [ ] VAE コードの実験管理を MLFlow に変更する。
- [ ] GAN コードを PyTorch Lighting 0.9.x 系に合わせる。
- [ ] GAN コードの実験管理を MLFlow に変更する。

## 参考資料

- 2017.12.4 [Loss function のあれこれ][37ma5ras]:
  autoencoder では BCEWithLogitsLoss が利用されるとのこと。
- 2019.12.25 [実務で使えるニューラルネットワークの最適化手法][acro-engineer]:
  SGD, Adam, AdamW, AdaBount, RAdam に関して比較実験を行っている。
  RAdam も比較的パラメータに頑健で扱いやすいとのこと。
- 2020.2.25 [MLflow の導入(2)実験の整理と比較を PyTorch+pytorch-lightning でやってみた][chowagiken]:
  PyTorch Lightning で MLFlow を使う場合の最小限の構成が書かれています。
- 2020.6.4 [Python: MLflow Tracking を使ってみる][cube_suger]:
  MLFlow の基本的な使い方が書かれています。
  Artifact URI がサーバから見える場所だけでなく、
  実験を実行する方からも見える必要があることが書かれています。
- 2020.6.26 [MLflow で実験管理入門][future]:
  MLFlow を利用した実験管理の方法について書かれています。
  ログの出力先として S3 の指定方法や、既存のエクセル管理があれば移行方法も書かれています。

[37ma5ras]: http://37ma5ras.blogspot.com/2017/12/loss-function.html
[acro-engineer]: http://acro-engineer.hatenablog.com/entry/2019/12/25/130000
[chowagiken]: https://blog.chowagiken.co.jp/entry/2020/02/25/MLflowの導入（２）実験の整理と比較をPyTorch%2Bpytorch-lightningでやっ
[cube_suger]: https://blog.amedama.jp/entry/mlflow-tracking
[future]: https://future-architect.github.io/articles/20200626/

### リポジトリ

- GitHub [AntixK/PyTorch-VAE][antixk]:
  PyTorch を利用して色々な種類の VAE を実装し、
  Celeba データセットで再構成結果とサンプリング結果を比較しています。
- GitHub [rasbt/deeplearning-models][rasbt]:
  基本的なネットワークアーキテクチャに関して pytorch で実装し、 ipynb 形式で公開している。

[antixk]: https://github.com/AntixK/PyTorch-VAE
[rasbt]: https://github.com/rasbt/deeplearning-models
