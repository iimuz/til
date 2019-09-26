# Learning Deep Features One-Class Classification (DOC)

[ディープラーニングを使った画像の異常検知　－論文と実装－][shinmura0] の再実装です。
元文献は [Learning Deep Features for One-Class Classification][arxiv] となります。

[arxiv]: https://arxiv.org/abs/1801.05365
[shinmura0]: https://qiita.com/shinmura0/items/cfb51f66b2d172f2403b

## Usage

実行は下記のコマンドで行います。

```sh
# 学習の実行
$ python train.py

# 認識の実行
$ python predict.py
```

## Tips

### ipynb

Google colaboratory で実行した結果を `doc_tensorflow.ipynb` として保存しています。

### CPU or GPU

Pipfile では、 CPU 版をインストールしています。
GPU 版を利用する場合は、 `tensorflow-gpu` をインストールしてください。

## ToDo

- モデルと学習などが一体となってしまっているので、分離作業を行う。
