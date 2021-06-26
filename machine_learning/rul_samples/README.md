# Remaining Usefule Life (RUL) samples

RUL 予測のサンプルです。
精度というよりも、利用できるライブラリやフレームワークを利用し簡易にモデル構築したメモを残しています。

| ライブラリ | データセット            | ノートブック                                        |
| :--------- | :---------------------- | :-------------------------------------------------- |
| None       | [PCoE No.6 FD001][pcoe] | [pcoe06_fd001_rul.ipynb][nb_pcoe06_fd001]           |
| LightGBM   | [PCoE No.6 FD001][pcoe] | [pcoe06_fd001_lightgbm.ipynb][nb_pcoe06_fd001_lgbm] |

[pcoe]: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
[nb_pcoe06_fd001]: pcoe06_fd001_rul.ipynb
[nb_pcoe06_fd001_lgbm]: pcoe06_fd001_lightgbm.ipynb

## 参考資料

(LightGBM + Optuna)

- [optuna.integration][optuna_integration]
- [LIghtGBM][lgbm]
  - [lightgbm.cv][lgbm_cv]
  - [lightgbm.Booster][lgbm_bosster]
  - [lightgbm.train][lgbm_train]
- 2020/01/20 [Optuna の拡張機能 LightGBM Tuner によるハイパーパラメータ自動最適化][pfn]
- 2020/06/01 [Python: Optuna の LightGBMTunerCV から学習済みモデルを取り出す][amedama]

[amedama]: https://blog.amedama.jp/entry/optuna-lightgbm-tunercv
[lgbm]: https://lightgbm.readthedocs.io/en/latest/index.html
[lgbm_cv]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
[lgbm_bosster]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster
[lgbm_train]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
[optuna_integration]: https://optuna.readthedocs.io/en/stable/reference/integration.html
[pfn]: https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/
