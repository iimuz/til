# AutoML

AutoML ライブラリを利用した機械学習モデルの作成サンプルです。
モデル生成を行うことを目的としているため、精度を上げるための処理などは実行していません。
データセットとライブラリ、実行したノートブックは下記のような対応関係となっています。

|       ライブラリ       |   データセット    |                   ノートブック                   |
| :--------------------: | :---------------: | :----------------------------------------------: |
|          None          | [PCoE No.6][pcoe] |        [pcoe06_turbofun.ipynb][nb_pcoe06]        |
| [AutoGluon][autogluon] | [PCoE No.6][pcoe] | [autogluon_pcoe06_turbofan][nb_autogluon_pcoe06] |

[autogluon]: https://auto.gluon.ai/stable/index.html
[nb_autogluon_pcoe06]: autogluon_pcoe06_turbofan.ipynb
[nb_pcoe06]: pcoe06_turbofun.ipynb
[pcoe]: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
