# Autoencoder using PyTorch

PyTorch を利用した Autoencoder 実装です。
対象データは、時系列データとしてます。

## 環境構築

Poetry で管理しているため、下記コマンドで環境構築できます。

```sh
poetry install
```

ただし、 pytorch 1.4.0 と torchvision, pytorch-lightning が依存関係ではじかれるため、
別に pytorch-lightning をインストールしてください。
