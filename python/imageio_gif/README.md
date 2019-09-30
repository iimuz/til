# imageio を利用した gif 画像の生成

imageio を利用して gif 画像を生成するサンプルです。
[Deep Convolutional Generative Adversarial Network][dcgan] を参考にしています。

[dcgan]: https://www.tensorflow.org/tutorials/generative/dcgan

## Usage

下記コマンドで環境構築と実際にサンプルコードを動作します。

```sh
pipenv install --skip-lock
pipenv run python create_gif.py
```

## 結果

結合した画像のサンプルです。

|          000          |          001          |          002          |          003          |          004          |
| :-------------------: | :-------------------: | :-------------------: | :-------------------: | :-------------------: |
| ![image000][image000] | ![image001][image001] | ![image002][image002] | ![image003][image003] | ![image004][image004] |

|          005          |          006          |          007          |          008          |          009          |
| :-------------------: | :-------------------: | :-------------------: | :-------------------: | :-------------------: |
| ![image005][image005] | ![image006][image006] | ![image007][image007] | ![image008][image008] | ![image009][image009] |

結合結果です。

| 結合したアニメーション画像 |
| :------------------------: |
|  ![animation image][gif]   |

[image000]: _docs/images000.png
[image001]: _docs/images001.png
[image002]: _docs/images002.png
[image003]: _docs/images003.png
[image004]: _docs/images004.png
[image005]: _docs/images005.png
[image006]: _docs/images006.png
[image007]: _docs/images007.png
[image008]: _docs/images008.png
[image009]: _docs/images009.png
[gif]: _docs/anim.gif
