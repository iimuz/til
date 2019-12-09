# Tips of matplotlib

matplotlib の使い方に関する簡単なサンプル集です。

## グラフ中の一部の背景色を変更

描画したグラフ中にの一部だけ背景色を変更するサンプルです。
例えば、時系列データの場合に、特定の日付からラベルが異なるような場合に、
ラベルごとに背景色を変更するといったことができます。

実行方法は下記のようになります。

```sh
pipenv run python span_background_color.py
```

出力結果は、 `data` フォルダに生成します。
縦方向または横方向に背景色を変更した場合に下記のような出力となります。

| 水平方向 | 垂直方向 |
| :-: | :-: |
| ![horizontal][fig_axhspan] | ![vertical][fig_axvspan] |

[fig_axhspan]: docs/axhspan.png
[fig_axvspan]: docs/axvspan.png

参考文献

- [matplotlibで一定区間に背景色をつける方法][bunseki-train]

[bunseki-train]: https://bunseki-train.com/axvspan-and-axhspan/

## グラフの背景に線を引く

描画したグラフの背景に線を引くサンプルです。
閾値の部分に線を引くなどができます。

実行方法は下記のようになります。

```sh
pipenv run python line_background.py
```

出力結果は、 `data` フォルダに生成します。
縦方向または横方向に線を引いた場合に下記のような出力となります。

| 水平方向 | 垂直方向 |
| :-: | :-: |
| ![horizontal][fig_axhline] | ![vertical][fig_axvline] |

[fig_axhline]: docs/axhline.png
[fig_axvline]: docs/axvline.png

## subplot の共通設定を一括で行う

subplot でグラフを描画した場合に、各 subplot の共通設定を一括で行う方法です。
`get_axes()` 関数を呼び出すことで、各 axes を取得することができます。

```py
for ax in plt.gcf().get_axes():
  plt.axes(ax)
```

参考文献

- [matplotlibのsubplotで共通の設定をオサレにやる][ceptree]

[ceptree]: https://qiita.com/ceptree/items/f52fab12bc07753f8909
