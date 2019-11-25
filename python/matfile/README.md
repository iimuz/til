# mat ファイルの読み込み

mat ファイルを python で取り扱うためのサンプルスクリプトです。

mat ファイルは、下記の形式に分かれるようです。

- v7.3 形式: [HDF5][hdfgroup] 形式となるため、 [h5py][h5py] ライブラリで読み込み可能
- v7.3 以前の形式: [scipy][scipy] ライブラリで読み込み可能

今回のサンプルでは、 v7.3 以前のファイルを対象とした読み込みを記載しています。

[h5py]: https://www.h5py.org/
[hdfgroup]: https://www.hdfgroup.org/solutions/hdf5/
[scipy]: https://www.scipy.org/

## Usage

下記コマンドで、圧縮及び解凍のスクリプトが実行できます。

```sh
# 必要なパッケージインストール
pipenv install

# mat ファイルのダウンロードと読み込みの実行
python -m unittest test_matfile
```

## 参考資料

- [Takuya Miyashita: mat ファイルからの読み込み][hydrocoast]

[hydrocoast]: https://hydrocoast.jp/index.php?Python/mat形式ファイルからの変数読み込み
