# Kaggle: IEEE-CIS Fraud Detection

Kaggle の [IEEE-CIS Fraud Detection][competition] に関するメモです。

```sh
kaggle competitions leaderboard ieee-fraud-detection -s
```

[competition]: https://www.kaggle.com/c/ieee-fraud-detection/

## Files

- `v0.1.0_plot_dataset.ipynb`: データセットを多くの加工をせずにプロットしています。
- `v0.1.0_plot_pivot_table.ipynb`: 週、月などで集約したデータをプロットしています。
- `v0.1.0_feature_engineering.ipynb`: 特徴量生成と LightGBM による簡易特徴量の重要度をプロットしています。
- `v0.1.0_lightgbm_optuna_tuner.ipynb`: Optuna の LightGBM Tuner を利用したパラメータ探。
