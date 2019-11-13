# Jupyter R

jupyter のカーネルとして R 環境を導入するサンプルです。
renv と pipenv を利用して環境を構築します。

## Usage

R Kernel を導入した jupyter 環境を起動するためには、下記のように実行します。

```sh
docker-compose run --rm jupyter-r bash

# in container
pipenv intall
pipenv run Rscript install_packages.R
exit

docker-compose up -d
```

## 参考資料

- [R を Jupyter Notebook で利用する][dividable]

[dividable]: https://dividable.net/programming/r-jupyter-notebook/
