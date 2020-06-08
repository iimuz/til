# 風力タービン高速ベアリングの経過予測

下記の記事における一部をサンプルとして python で実装しています。

- MathWorks: [風力タービン高速ベアリングの経過予測][wind]

[wind]: https://jp.mathworks.com/help/predmaint/ug/wind-turbine-high-speed-bearing-prognosis.html

## 注意点

- 指数劣化モデルによる予測は実装していません。
- 周波数領域における特徴量が文献の結果と一致しません。
- PCA結果の健康状態の正負が反転しています。

## Usage

一連の結果を得るために必要なコマンドを記載します。

```sh
pipenv install --skip-lock  # 必要なパッケージのインストール
pipenv run python -m unittest test.test_download  # データセットのダウンロード
pipenv run python -m unittest test.test_dataset   # .mat ファイルを読み込む
```

特徴量化などに関しては、 notebook フォルダの `show_data.ipynb` を参照してください。

