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
