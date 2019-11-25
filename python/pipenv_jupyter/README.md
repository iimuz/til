# pipenv で jupyter 環境を構築するサンプル

pipenv で jupyter 環境を構築します。
また、 vscode で操作できることも確認しています。

```sh
pipenv install --skip-lock jupyter
pipenv run jupyer notebook --config jupyter_notebook_config.py
```
