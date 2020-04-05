# Kaggle: Store Item Demand Forecasting Challenge

Kaggle の [Store Item Demand Forecasting Challenge][competition] に関するメモです。

```sh
kaggle competitions leaderboard demand-forecasting-kernels-only -s
```

[competition]: https://www.kaggle.com/c/demand-forecasting-kernels-only/data

## Files

- `v0.1.0_plot_dataset.ipynb`: データセットを多くの加工をせずにプロットしています。
- `v0.1.0_plot_pivot_table.ipynb`: 週、月などで集約したデータをプロットしています。
- `v0.1.0_feature_engineering.ipynb`: 特徴量生成と LightGBM による簡易特徴量の重要度をプロットしています。
- `v0.1.0_lightgbm_optuna_tuner.ipynb`: Optuna の LightGBM Tuner を利用したパラメータ探。
