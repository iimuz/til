# minimal python

python のコードを書くときに利用する最小限の設定です。
ただし、ここでの対象は実行用スクリプトを開発することを想定しています。
そのため、下記は対象外です。

- pip インストール可能なライブラリ開発
- pip インストール可能なツール開発

## ファイル構成

- `.gitignore`: [python 用の gitignore][link04] です。
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

- [black][link00]: python code formmater.
- [flake8][link01]: style checker.
- [isort][link02]: sort imports.
- [mypy][link03]: static typing.
- docstirng: [numpy 形式][link05]を想定しています。
  - vscodeの場合は [autodocstring][link06] 拡張機能によりひな型を自動生成できます。

[link00]: https://github.com/psf/black
[link01]: https://github.com/PyCQA/flake8
[link02]: https://github.com/PyCQA/isort
[link03]: https://github.com/python/mypy
[link04]: https://github.com/github/gitignore/blob/main/Python.gitignore
[link05]: https://numpydoc.readthedocs.io/en/latest/format.html
[link06]: https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring
