# Altair + React.js

バックエンドにAltairを利用し、フロントエンドにReact.jsを利用した場合に、
Altairでの描画結果をReact.jsで描画する方法の試作。

## 構成

- backend
- frontend

## backend

下記コマンドなどの利用を想定しています。
(仮想環境は有効化されていると想定したコマンドとなります。)

- `uvicorn src.main:app --reload`: ライブサーバーの起動
  - `http://localhost:8000`: ブラウザでの基本アクセス
  - `http://localhost:8000/docs`: Swagger UIへのアクセス

### 仮想環境の構築

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

### code style

コードの整形などはは下記を利用しています。

- [black](https://github.com/psf/black): python code formmater.
- [flake8](https://github.com/PyCQA/flake8): style checker.
- [isort](https://github.com/PyCQA/isort): sort imports.
- [mypy](https://github.com/python/mypy): static typing.
- docstirng: [numpy 形式](https://numpydoc.readthedocs.io/en/latest/format.html)を想定しています。
  - vscodeの場合は下記の拡張機能によりひな型を自動生成できます。
    - [autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)

## frontend

Webpack + React + Typescript を利用した環境。

- `.eslintrc.json`: eslint 用の設定
- `.prettierrc.json`: prettier 用の設定

下記のコマンドが利用できるように設定している。

- `npm run build`: buildディレクトリ以下にファイル群を作成
- `npm run format`: ファイルの整形チェックを行い修正
- `npm run lint`: ファイルの整形チェック
- `npm run production`: 最終成果物用のビルド
- `npm run start`: サーバーを起動し、ファイル変更を検知してリロード
