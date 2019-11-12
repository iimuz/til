# Autoencoder using tensorflow

単純な Fully Connected 層だけの Autoencoder の実装サンプルです。

## Usage

簡易な学習を実行する場合は、下記のようにして実行します。

```sh
pipenv install --skip-lock
pipenv run python -m unittest test.test_dense_ae
```

各モデルに対してテストコードを用意しています。
それぞれのモデルを実行するにはテスト実行の対象コードを変更してください。

## Models

下記にモデルファイル名と簡易な説明を記載します。

- `src.models.cvae`: Convolutional Variational Autoencoder
- `src.models.dense_ae`: 単純な Dense 層のみで構成した Autoencoder
- `src.models.simple_ae`: 非常にシンプルな Convolutional Autoencoder
