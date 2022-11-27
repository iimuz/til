# MT4 csv to DB

[MT4 の履歴を csv 化するスクリプト](https://github.com/iimuz/til/src/til-20221106/)で出力した csv を DB に取り込むためのスクリプトです。

## 実行方法

下記の手順でデフォルトの引数で実行ができます。

1. `.sample.env` を元に `.env` ファイルを作成
1. docker compose を利用して実行

   ```sh
   docker compose build
   docker compose run --rm -it app
   ```

## ファイル構成

- `.gitignore`: [python 用の gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore) です。
- `.sample.env`: 環境変数のサンプルを記載します。利用時は `.env` に変更して利用します。
- `LICENSE`: ライセンスを記載します。 MIT ライセンスを設定しています。
- `setup.py`/`setup.cfg`: python バージョンなどを明記します。
- `requirements.txt`: 利用するパッケージを記述します。
- `README.md`: 本ドキュメントです。
- `.vscode`: VSCode の基本設定を記述します。
- `.devcontainer`: VSCode Remote Containers の設定を記述します。
- `src`: 開発するスクリプトを格納します。

## 環境変数

下記の環境変数を利用します。

- `USER_UID`, `USER_GID`: VSCode remote development の docker 環境で利用するユーザ ID とグループ ID。
- `TIMEZONE_HOURS`: csv から読み込んだ時刻のタイムゾーンを特定するための UTC からの時差。
  - MT4 の履歴取得ではタイムゾーンを付けた表記になっていないため、タイムゾーンを付与する。ただし、保存時は UTC に補正して保存する。

## 仮想環境の構築

仮想環境の構築には python 標準で付属している venv の利用を想定しています。
スクリプトで必要なパッケージは `requirements.txt` に記載します。
実際にインストール後は、 `requirements-freeze.txt` としてバージョンを固定します。

```sh
# create virtual env
python -m venv .venv

# activate virtual env(linux)
source .venv/bin/activate
# or (windows)
.venv/Scripts/activate.ps1

# install packages and freeze version
pip install -r requirements.txt
pip freeze > requirements-freeze.txt

# recreate virtual env
pip install -r requirements-freeze.txt
```

## code style

コードの整形などは下記を利用しています。

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
また、共通設定ではなく個人設定として、docker 環境にマウントするディレクトリを増やす場合などは、 `.devcontainer/docker-composel.local.yml` を編集します。
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
