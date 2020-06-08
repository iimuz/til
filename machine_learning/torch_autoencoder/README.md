# Autoencoder using PyTorch

PyTorch を利用した Autoencoder 実装です。
対象データは、時系列データとしてます。

## 環境構築

Poetry で管理しているため、下記コマンドで環境構築できます。

```sh
poetry install
```

## Usage

学習は下記のように実行します。

```sh
python -m src.models.train SimpleAE
```

## 参考資料

- 2018.2.22 [Pytorch による AutoEncoder Family の実装][dl_kento]
  - とても浅い CNN が掲載されています。
- 2019.9.28 [LSTM for time series prediction][de8aeb26f2ca]
  - vwap データセットの作成方法が記載してあります。

[de8aeb26f2ca]: https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca
[dl_kento]: http://dl-kento.hatenablog.com/entry/2018/02/22/200811
