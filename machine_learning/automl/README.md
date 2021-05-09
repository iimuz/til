# AutoML

AutoML ライブラリを利用した機械学習モデルの作成サンプルです。
モデル生成を行うことを目的としているため、精度を上げるための処理などは実行していません。
データセットとライブラリ、実行したノートブックは下記のような対応関係となっています。

| ライブラリ             | データセット      | ノートブック                                           |
| :--------------------- | :---------------- | :----------------------------------------------------- |
| None                   | [PCoE No.6][pcoe] | [pcoe06_turbofun.ipynb][nb_pcoe06]                     |
| [AutoGluon][autogluon] | [PCoE No.6][pcoe] | [autogluon_pcoe06_turbofan.ipynb][nb_autogluon_pcoe06] |
| [AutoKeras][autokeras] | [PCoE No.6][pcoe] | [autokeras_pcoe06_turbofan.ipynb][nb_autokeras_pcoe06] |
| [PyCaret][pycaret]     | [PCoE No.6][pcoe] | [pycaret_pcoe06_turbofan.ipynb][nb_pycaret_pcoe06]     |
| [TPOT][tpot]           | [PCoE No.6][pcoe] | [tpot_pcoe06_turbofan.ipynb][nb_tpot_pcoe06]           |

各ライブラリに関する簡単な比較表を下記に記載します。
簡単に調べた範囲なので、間違っていること、抜けている可能性はあります。

| ライブラリ | Classification | Regression | Clustering | Anomaly Detection | 対象アルゴリズムの例       | feature importance |
| :--------- | :------------: | :--------: | :--------- | :---------------: | -------------------------- | ------------------ |
| AutoGluon  |       o        |     o      | x          |         x         | sklearn, LightGBM, NN      | o                  |
| AutoKeras  |       o        |     o      | x          |         x         | NN                         | x                  |
| PyCaret    |       o        |     o      | o          |         o         | sklearn, LightGBM, XGBoost | △(Tree 系のみ)     |
| TPOT       |       o        |     o      | x          |         x         | sklearn, XGBoost, NN       | o                  |

- AutoKeras: keras バックエンドでネットワークの最適化を行ってくれる。
- AutoGluon: sklearn, LightGBM, NN などのモデルの最適化を実行する。
- PyCaret: sklearn, LightGBM, XGBoost などのモデルを最適化する。Clustering と Anomaly Detection にも対応している。
- TPOT: sklearn と XGBoost などのモデルを GA(Genetic Algorithm)を用いて最適化する。

[autokeras]: https://autokeras.com/
[autogluon]: https://auto.gluon.ai/stable/index.html
[nb_autogluon_pcoe06]: autogluon_pcoe06_turbofan.ipynb
[nb_autokeras_pcoe06]: autokeras_pcoe06_turbofan.ipynb
[nb_pcoe06]: pcoe06_turbofun.ipynb
[nb_tpot_pcoe06]: tpot_pcoe06_turbofan.ipynb
[nb_pycaret_pcoe06]: pycaret_pcoe06_turbofan.ipynb
[pcoe]: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
[pycaret]: https://pycaret.readthedocs.io/en/latest/index.html
[tpot]: http://epistasislab.github.io/tpot/
