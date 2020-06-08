# Time Series Forecasting

Time Series に対する予測モデルの実装集です。

## Dataset

データセットとして、多変量時系列データとなる `jena_climate_2009_2016` を利用しています。
読み込んだデータセットの基礎情報及び、先頭、末尾を pandas で表示した結果を下記に示します。

```txt
RangeIndex: 420551 entries, 0 to 420550
Data columns (total 15 columns):
 #   Column           Non-Null Count   Dtype
---  ------           --------------   -----
 0   Date Time        420551 non-null  object
 1   p (mbar)         420551 non-null  float64
 2   T (degC)         420551 non-null  float64
 3   Tpot (K)         420551 non-null  float64
 4   Tdew (degC)      420551 non-null  float64
 5   rh (%)           420551 non-null  float64
 6   VPmax (mbar)     420551 non-null  float64
 7   VPact (mbar)     420551 non-null  float64
 8   VPdef (mbar)     420551 non-null  float64
 9   sh (g/kg)        420551 non-null  float64
 10  H2OC (mmol/mol)  420551 non-null  float64
 11  rho (g/m**3)     420551 non-null  float64
 12  wv (m/s)         420551 non-null  float64
 13  max. wv (m/s)    420551 non-null  float64
 14  wd (deg)         420551 non-null  float64
dtypes: float64(14), object(1)
memory usage: 48.1+ MB

Date Time  p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
0  01.01.2009 00:10:00    996.52     -8.02    265.40        -8.90    93.3          3.33          3.11          0.22       1.94             3.12       1307.75      1.03           1.75     152.3
1  01.01.2009 00:20:00    996.57     -8.41    265.01        -9.28    93.4          3.23          3.02          0.21       1.89             3.03       1309.80      0.72           1.50     136.1
2  01.01.2009 00:30:00    996.53     -8.51    264.91        -9.31    93.9          3.21          3.01          0.20       1.88             3.02       1310.24      0.19           0.63     171.6
3  01.01.2009 00:40:00    996.51     -8.31    265.12        -9.07    94.2          3.26          3.07          0.19       1.92             3.08       1309.19      0.34           0.50     198.0
4  01.01.2009 00:50:00    996.51     -8.27    265.15        -9.04    94.1          3.27          3.08          0.19       1.92             3.09       1309.00      0.32           0.63     214.3

Date Time  p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
420546  31.12.2016 23:20:00   1000.07     -4.05    269.10        -8.13   73.10          4.52          3.30          1.22       2.06             3.30       1292.98      0.67           1.52     240.0
420547  31.12.2016 23:30:00    999.93     -3.35    269.81        -8.06   69.71          4.77          3.32          1.44       2.07             3.32       1289.44      1.14           1.92     234.3
420548  31.12.2016 23:40:00    999.82     -3.16    270.01        -8.21   67.91          4.84          3.28          1.55       2.05             3.28       1288.39      1.08           2.00     215.2
420549  31.12.2016 23:50:00    999.81     -4.23    268.94        -8.53   71.80          4.46          3.20          1.26       1.99             3.20       1293.56      1.49           2.16     225.8
420550  01.01.2017 00:00:00    999.82     -4.82    268.36        -8.42   75.70          4.27          3.23          1.04       2.01             3.23       1296.38      1.23           1.96     184.9
```

## 参考資料

- TensorFlow [Time series forecasting][tf_time_series]

[tf_time_series]: https://www.tensorflow.org/tutorials/structured_data/time_series
