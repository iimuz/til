# unittest のサンプル

unittest の簡易サンプルです。

## テストの実行方法

tests 下の全フォルダを実行する。
`discover` を付与してフォルダを指定する。

```sh
$ python -m unittest discover tests
```

単一のテストのみを実行する場合は、モジュールを指定する。

```sh
$ python -m unittest tests.test_hoge
```

テスト結果の詳細を取得したい場合は、 `-v` オプションを付与する。

```sh
$ python -m unittest discover tests -v
test_one (test_hoge.TestHoge) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

テストフォルダを再帰的に `discover` で一括で行いたい場合は、
テストフォルダに `__init__.py` を配置することで `discover` コマンドが探索します。
ソースフォルダに配置する必要はありません。

```sh
$ touch tests/__init__.py
$ touch tests/test_subsample/__init__.py
$ python -m unittest discover tests -v
test_one (test_hoge.TestHoge) ... ok
test_one (test_sabsample.test_geho.TestGeho) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK
```
