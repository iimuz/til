# AutoML

AutoML ライブラリを利用した機械学習モデルの作成サンプルです。
モデル生成を行うことを目的としているため、精度を上げるための処理などは実行していません。
データセットとライブラリ、実行したノートブックは下記のような対応関係となっています。

| ライブラリ             | データセット      | ノートブック                                           |
| :--------------------- | :---------------- | :----------------------------------------------------- |
| None                   | [PCoE No.6][pcoe] | [pcoe06_turbofun.ipynb][nb_pcoe06]                     |
| [AutoGluon][autogluon] | [PCoE No.6][pcoe] | [autogluon_pcoe06_turbofan.ipynb][nb_autogluon_pcoe06] |
| [TPOT][tpot]           | [PCoE No.6][pcoe] | [tpot_pcoe06_turbofan.ipynb][nb_tpot_pcoe06]           |

各ライブラリに関する簡単な比較表を下記に記載します。
簡単に調べた範囲なので、間違っていること、抜けている可能性はあります。

| ライブラリ | Classification | Regression | 対象アルゴリズムの例  | feature importance |
| :--------- | :------------: | :--------: | :-------------------- | :----------------: |
| AutoGluon  |       o        |     o      | sklearn, LightGBM, NN |         o          |
| TPOT       |       o        |     o      | sklearn, XGBoost, NN  |         o          |

- AutoGluon: sklearn, LightGBM, NN などのモデルの最適化を実行する。
- TPOT: sklearn と XGBoost などのモデルを GA(Genetic Algorithm)を用いて最適化する。

[autogluon]: https://auto.gluon.ai/stable/index.html
[nb_autogluon_pcoe06]: autogluon_pcoe06_turbofan.ipynb
[nb_pcoe06]: pcoe06_turbofun.ipynb
[nb_tpot_pcoe06]: tpot_pcoe06_turbofan.ipynb
[pcoe]: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
[tpot]: http://epistasislab.github.io/tpot/
