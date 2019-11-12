# renv sample

renv を利用してパッケージ管理を行う方法の簡単なサンプルです。

## Usage

### 管理の開始

下記は全て R のインタラクティブシェル内で行います。

1. 初期化: `renv::init()`
1. パッケージの追加: `renv::install("hoge")`
1. 状態の保存: `renv::snapshot()`

git などで管理する対象としては下記のファイルになります。

- `.Rprofile`
- `renv.lock`
- `renv/activate.R`

renv ディレクトリ内の他のファイルは管理する必要はないようです。

### 状態の復元

git clone などで初期化した状態から復元するには下記のコマンドを実行します。

- `renv::restore()`

## renv のインストール

renv 自体のインストールは下記のようにして実行します。

```R
if (!requireNamespace("remotes"))
  install.packages("remotes")

remotes::install_github("rstudio/renv")
```

## docker 環境へのインストール

docker コンテナに renv をインストールする場合は下記のコマンドを追加します。

```Dockerfile
ENV RENV_VERSION 0.5.0-25
RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"
```

`RENV_VERSION` を指定しない場合は、 master からインストールすることになります。

## 参考資料

- [GitHub: rstudio/renv][renv]
- [renv: Introduction to renv][renv-introduction]
- [renv: Using renv with Docker][renv-docker]
- 2019.8.16: [R のパッケージ管理のための renv を使ってみた][okiyuki99]
- 2019.9.28: [renv によるパッケージ管理][black_tank_top]

[black_tank_top]: https://speakerdeck.com/black_tank_top/renv-version-control
[okiyuki99]: https://qiita.com/okiyuki99/items/688a00ca9a58e42e3bfa
[renv]: https://github.com/rstudio/renv
[renv-docker]: https://rstudio.github.io/renv/articles/docker.html
[renv-introduction]: https://rstudio.github.io/renv/articles/renv.html
