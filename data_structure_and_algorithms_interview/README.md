# 50+ Data Structure and Algorithms Interview Questions for Programmers の挑戦

[50+ Data Structure and Algorithms Interview Questions for Programmers][hackernoon] の記事の問題を解いた時のメモです。

* [日本語訳][postd]

[hackernoon]: https://hackernoon.com/50-data-structure-and-algorithms-interview-questions-for-programmers-b4b1ac61f5b0
[postd]: https://postd.cc/50-data-structure-and-algorithms-interview-questions-for-programmers/

## テスト実行

各問題に対する解答と実際に動かすコードは別にしています。
動作確認はテストコードを用いて行うため、下記のようにして実行します。

### python

* 全件テスト

  ```sh
  $ make test
  # or
  $ python -m unittest discover
  ```
* 特定のテスト

  ```sh
  $ make test/problem002
  # or
  $ python -m unittest problem001/test_main.py
  ```

### c++

c++ の方は全件を同時に実行するコマンドは用意していません。

* gtest のビルド

  ```sh
  $ make build/gtest
  ```
* テストのビルドと実行

  ```sh
  $ make build/problem010
  $ make run/problem010
  ```

## Table of Content

* problem001: 1 から 100 までの与えられた整数配列の中から足りない数字を探す。
* problem002: 与えられた整数の配列において重複した数字を探す。
* problem003: ソートされていない配列から最大値と最小値を探す。
  * ライブラリや組み込み関数を利用せずに実現する。
    python の場合は、 list に対して max や min が取得できる関数を使わない。
* problem004: 整数の配列と合計値が与えられたときに、整数の配列の中の 2 つの要素の組み合わせで、
  合計値と一致するパターンを全て探す。
* problem005: problem006 と同じような問題で、回答へのリンク先が問題と異なっているように見えるので飛ばす。
* problem006: 整数の配列内に複数の重複がある場合に、重複を削除するにはどうすればよいか。
  * set などの Collection API を利用しない。
* problem007: クイックソートを実装せよ。
* problem008: problem006 との違いが判らないので飛ばす。
* problem009: リストを反転せよ。
* problem010: ライブラリを全く使わないで重複を削除する。
  * problem006 でほぼ使っていないので飛ばす。
    これ以上は、 c++ の方が書きやすい。
* problem010:
