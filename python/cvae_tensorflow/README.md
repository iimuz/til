# Convolutional Variational Auto Encoder using tensorflow

[Convolutional Variational Autoencoder][tutorial] の写経です。

[tutorial]: https://www.tensorflow.org/tutorials/generative/cvae

## Usage

```sh
python dataset.py  # データセットの確認
python network.py  # ネットワークの確認
python train.py    # 学習処理の実行
```

## Tips

### .ipynb

Google Colaboratory で実行した結果が、 `cvae_tensorflow.py` となります。

### VAE における Encoder

VAE において Encoder は、潜在空間を表現する分布を出力とする。
そして、 Encoder が生成した分布からサンプリングした値を Decoder の入力とする。
下記の図が分かりやすかった。

| [人工知能に関する断創録][aidiary] より引用 |
| :----------------------------------------: |
|     ![CVAE Architecture][aidiary-arch]     |

[aidiary]: https://aidiary.hatenablog.com/entry/20180228/1519828344
[aidiary-arch]: https://cdn-ak.f.st-hatena.com/images/fotolife/a/aidiary/20180228/20180228212323.png

### 損失関数

Encoder は、入力した値から分布を予測することを目的とする。
今、 Encoder の予測する分布を $P(z|x)$ とし、真の分布を $Q(z|x)$ とする。
そうすると、二つの分布間の距離を KL divergence $D_{KL}(Q(z|x) || P(z|x))$ で最小化する問題になる。

### Reparametrize Trick

Reparametrize Trick は、 Encoder が推定した分布からサンプリングした値を Decodoer の入力とすると、
途中で伝搬が途切れるということになる問題への対応です。
説明としては、下記の図が分かりやすかった。

| 推定した分布からサンプリングするときの伝搬 |    Reparametrize Trick    |
| :----------------------------------------: | :-----------------------: |
|   ![split forwarding][split-forwarding]    | ![forwarding][forwarding] |

[Variational Autoencoders (VAEs; 変分オートエンコーダ) (理論)][cympfh] より引用

潜在空間の分布を正規分布で仮定すると、推定した分布が下記のようになる。

```math
N(\mu, \Sigma) = \mu + \Sigma^2 N(0, I)
```

つまり、 $N(0, I)$ からサンプリングした値を利用して
$N(\mu, \Sigma)$ からサンプリングした値に変換する。
$N(0, I)$ の方はランダムにサンプリングしておいて、伝搬できる必要がない。
結果として、 $N(\mu, \Sigma)$ の方は伝搬できる状態になるので良いということのようです。

[cympfh]: https://qiita.com/cympfh/items/50b19933fd3834e86862
[forwarding]: https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F17022%2F44667e70-e4c4-3cee-02c1-f2f4329c1d56.png?ixlib=rb-1.2.2&auto=compress%2Cformat&gif-q=60&s=2c399576b7094004802704b9a82c97b1
[split-forwarding]: https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F17022%2Fabc73e5c-a2ac-2035-b44a-e01855f66213.png?ixlib=rb-1.2.2&auto=compress%2Cformat&gif-q=60&s=765519dfe6fe8432031d3882a2e117c0
