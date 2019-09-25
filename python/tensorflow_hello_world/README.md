# hello tensorflow

tensorflow の環境と hello world の実行確認のサンプルです。

## Usage

下記コマンドで実行できます。
実行環境は、 CPU となっています。

```sh
$ python hello.py

2019-09-24 12:27:16.434002: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-09-24 12:27:16.441488: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2712000000 Hz
2019-09-24 12:27:16.441973: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x2052cd0 executing computations on platform Host. Devices:
2019-09-24 12:27:16.443042: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
b'hello world'
```

docker を利用する場合は、下記コマンドで実行できます。

```sh
$ docker-compose run --rm dev
```

## Tips

### GPU

gpu バージョンを利用する場合は、 Pipfile のパッケージを変える必要があります。
現時点では、 [GPU と CPU 環境でインストールするパッケージを切り替えることはできない][halhorn] ようです。

- tensorflow -> tensorflow-gpu

また、 docker のイメージを GPU 版に帰る場合は、下記のイメージに変更してください。

- tensorflow/tensorflow:1.13.2-py3 -> tensorflow/tensorflow:1.13.2-gpu-py3

[halhorn]: https://qiita.com/halhorn/items/2fba53cf65e994b7de76

### ipynb

[Collaboratory][colab] を利用して実行した結果が `hello.ipynb` になります。

[colab]: https://colab.research.google.com/

