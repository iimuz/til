# Import bank statement csv

銀行口座の明細書をcsvダウンロードした後にデータベースに取り込むためのスクリプトです。
csvダウンロードは自動化していないため手動でダウンロードする必要があります。

## ファイル構成

- `.gitignore`: [python 用の gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore) です。
- `.sample.env`: 環境変数のサンプルを記載します。利用時は `.env` に変更して利用します。
- `LICENSE`: ライセンスを記載します。 MIT ライセンスを設定しています。
- `setup.py`/`setup.cfg`: python バージョンなどを明記します。
- `requirements.txt`: 利用するパッケージを記述します。
- `README.md`: 本ドキュメントです。
- `.vscode`: VSCodeの基本設定を記述します。
- `src`: 開発するスクリプトを格納します。

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
source .venv/Scripts/activate.ps1

# install packages and freeze version
pip install -r requirements.txt
pip freeze > requirements-freeze.txt

# recreate virtual env
pip install -r requirements-freeze.txt
```

## code style

コードの整形などはは下記を利用しています。

- [black](https://github.com/psf/black): python code formmater.
- [flake8](https://github.com/PyCQA/flake8): style checker.
- [isort](https://github.com/PyCQA/isort): sort imports.
- [mypy](https://github.com/python/mypy): static typing.
- docstirng: [numpy 形式](https://numpydoc.readthedocs.io/en/latest/format.html)を想定しています。
  - vscodeの場合は [autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) 拡張機能によりひな型を自動生成できます。

## VSCode環境

Remote dev container環境を利用して起動することができます。
リポジトリのコード実行に最低限必要な環境を `Dockerfile`, `docker-compose.yml` で定義しています。そのため、VSCode用に必要な追加設定については、 `docker-compose.extend.yml` に記載しています。

ただし、個人特有のリポジトリにコミットすることなくパラメータを変更する場合は、 `docker-compose.local.yml` を編集します。
ただし、変更を追跡しないようにするためには下記のように設定します。

```sh
git update-index --assume-unchanged .devcontainer/docker-compose.local.yml
```

例えば、 `worktree` などを利用している場合は、本リポジトリのみをマウントしてもgitが有効にならないため、 `docker-compose.local.yml` に下記のような定義を追加して対応できます。

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
