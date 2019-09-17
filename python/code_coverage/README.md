# code coverage

python でテスト時のコードカバレッジを計測するサンプルコードです。

## Usage

対象のテストを全て実行する場合は、
unittest の discover を利用して下記のように実行します。

```sh
$ coverage run -m unittest discover .
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

結果を目視で確認する場合は、 report コマンドを利用します。

```sh
$ coverage report
Name             Stmts   Miss  Cover
------------------------------------
sample.py            2      0   100%
test_sample.py       5      0   100%
------------------------------------
TOTAL                7      0   100%
```

単にファイルで計測した結果の通過場所を確認したい場合は、 annotate コマンドを利用します。
.cover ファイルが出来上がり、通過していない行にエクスクラメーションマークがつきます。

```sh
$ coverage annotate
```

html で出力したい場合は、 html コマンドを利用します。

```sh
$ coverage html
```

最終的に、結果を削除する場合は erase コマンドを利用します。
ただし、 html のフォルダなどは消えませんでした。

```sh
$ coverage erase
```
