---
title: epub to text
date: 2023-04-09
lastmod: 2023-04-09
---

## 概要

epub からテキストを抽出するためのスクリプトです。

## ファイル構成

- フォルダ
  - `.devcontainer`: VSCode Remote Containers の設定を記述します。
  - `.vscode`: VSCode の基本設定を記述します。
  - `src`: 開発するスクリプトを格納します。
- ファイル
  - `.gitignore`: [python 用の gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore) です。
  - `.sample.env`: 環境変数のサンプルを記載します。利用時は `.env` に変更して利用します。
  - `LICENSE`: ライセンスを記載します。 MIT ライセンスを設定しています。
  - `pyproject.toml`/`setup.py`/`setup.cfg`: python バージョンなどを明記します。
  - `README.md`: 本ドキュメントです。

## 実行方法

コマンドのみを実行できれば良い場合は下記のように実行します。

```sh
docker compose run --rm -it app
```

vscode と同様の開発環境で起動する場合は下記のように実行します。

```sh
docker compose -f docker-compose.yml -f .devcontainer/docker-compose.extend.yml -f .devcontainer/docker-compose.local.yml run --rm -it app
```

## 仮想環境の構築

仮想環境の構築には python 標準で付属している venv の利用を想定しています。
スクリプトで必要なパッケージは `pyprojec.toml` に記載します。

```sh
# create virtual env
python -m venv .venv

# activate virtual env(linux)
source .venv/bin/activate
# or (windows)
.venv/Scripts/activate

# install packages
pip install -e .[dev,test]
```

## code style

コードの整形などはは下記を利用しています。

- [black](https://github.com/psf/black): python code formmater.
- [flake8](https://github.com/PyCQA/flake8): style checker.
- [isort](https://github.com/PyCQA/isort): sort imports.
- [mypy](https://github.com/python/mypy): static typing.
- docstirng: [numpy 形式](https://numpydoc.readthedocs.io/en/latest/format.html)を想定しています。
  - vscode の場合は [autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) 拡張機能によりひな型を自動生成できます。

## VSCode remote containers

VSCode のリモートコンテナを利用した環境構築を利用できます。
下記拡張機能をインストールし、 `Dev Contianers: Reopen in Container` コマンドを実行することで Dockerfile の環境で VSCode を利用することができます。

- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

Container 環境にユーザを作成するため環境変数に下記の設定を追加してください。環境変数がない場合は、ユーザ名`vscode`、`UID=1000`, `GID=1000`でユーザを作成します。

- `USER_UID`: コンテナ内で利用するユーザの ID
- `USER_GID`: コンテナ内で利用するユーザのグループ ID

リポジトリのコード実行に最低限必要な環境を `Dockerfile`, `docker-compose.yml` で定義しています。そのため、VSCode 用に必要な追加設定については、 `.devcontainer/docker-compose.extend.yml` に記載します。
また、共通設定ではなく個人設定として、docker 環境にマウントするディレクトリを増やす場合などは、 `.devcontainer/docker-compose.local.yml` を編集します。
編集前に git で変更を検知しないように下記の設定を行ってください。下記の設定を行うことで、変更をリポジトリにコミットしないで修正可能になります。

```sh
git update-index --skip-worktree .devcontainer/docker-compose.local.yml
```

例えば、 `git worktree` などを利用している場合は、本リポジトリのみをマウントしても git が有効にならないため、 `docker-compose.local.yml` に下記のような定義を追加して対応できます。

```yml
services:
  app:
    volumes:
      - type: bind
        source: $HOME/path/to/git/worktree/root
        target: $HOME/path/to/git/worktree/root
      - type: bind
        source: $PWD
        target: $PWD
```

python 仮想環境を named volume で用意しています。WSL で実行する場合に、named volume をマウントした位置のホスト側に root 権限で空のフォルダが作成されます。そのため、先に `mkdir .venv` などでフォルダを作成しておくことで root 権限でのフォルダ生成を回避することができます。
