# Python での簡易サンプルコード

python で調べたり、簡易のコードを記述したときの実行サンプルのメモです。
各フォルダに 1 つのサンプルを配置しています。
サンプルの詳細は、各フォルダの README.md を参照してください。

## サンプルリスト

- code_coverage: コードカバレッジの計測用サンプル
- cvae_tensorflow: CVAE を tensorflow で記述したサンプル
- datasets: データセットのダウンロード用スクリプト
- dcgan_tensorflow: DCGAN を tensorflow で記述したサンプル
- depwalker: dependency walker の依存関係結果からネットワークグラフを表示するサンプル
- doc_tensorflow: DOC を tensorflow で記述したサンプル
- image-resize: 画像をリサイズするサンプル
- image_concat: 複数の画像を並べて 1 枚の画像とするサンプル
- imageio_gif: imageio パッケージを利用して gif アニメーションファイルを作成するサンプル
- overlap_diff_image: 基準画像との差異を基準画像にオーバーラップ表示するサンプル
- tensorflow-eager-mnist: tensorflow の eager mode で MNIST を分類するサンプル
- tensorflow-transfer-learning: tensorflow で転移学習するサンプル
- tensorflow_hello_world: tensorflow の hello world サンプル
- unittest_sample: unittest を実行するサンプル
- view_bokeh: bokeh を利用してブラウザにインタラクティブなグラフ表示を行うサンプル
- view_bokeh_table: bokeh を利用してブラウザで csv をテーブル表示するサンプル
- view_bokeh_network: bokeh と networkx を利用したネットワークグラフを表示するサンプル
- view_bokeh_network_attribute: bokeh と networkx を利用したネットワークグラフを表示するサンプルでノードとエッジ情報を付与したサンプル

## 開発環境

### pipfile

各フォルダに pipfile を用意しています。
フォルダ内で `pipenv install` で必要な環境が構築できます。

### vscode

docker コンテナ内で開発しています。
開発用 docker コンテナは vscode で利用するため、 `.devcontainer` に含まれています。
