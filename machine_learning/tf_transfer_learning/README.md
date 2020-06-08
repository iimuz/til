# transfer learning using tensorflow

[Transfer learning with a pretrained ConvNet][tutorials] の写経です。

[tutorials]: https://www.tensorflow.org/tutorials/images/transfer_learning

## Usage

```sh
$ python datasts.py  # データセットの確認

$ python network_fe.py  # Feature Extraction 用ネットワークの確認
$ python feature_extraction.py  # Feature Extraction の実行

$ python network_ft.py  # Fine Tuning 用ネットワークの確認
$ python fine_tuning.py  # Fine Tuning の実行
```

## Tips

### ipynb

Google Colaboratory で実行した結果は、 `tensorflow_transfer_learning.ipynb` となります。
