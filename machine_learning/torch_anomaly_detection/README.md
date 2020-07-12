# Anomaly Detection

## Tensorboard のインストールと起動

Tensorboard を poetry install で入れようとすると失敗します。
下記の手順で pip と setuptools を更新する必要があります。

```sh
poetry run pip install --update pip
poetry run pip install setuptools --upgrade
```

## 参考資料

- 2017.12.4 [Loss function のあれこれ][37ma5ras]
  - autoencoder では BCEWithLogitsLoss が利用されるとのこと。
- 2019.12.25 [実務で使えるニューラルネットワークの最適化手法][acro-engineer]
  - SGD, Adam, AdamW, AdaBount, RAdam に関して比較実験を行っている。
  - RAdam も比較的パラメータに頑健で扱いやすいとのこと。

[37ma5ras]: http://37ma5ras.blogspot.com/2017/12/loss-function.html
[acro-engineer]: http://acro-engineer.hatenablog.com/entry/2019/12/25/130000
