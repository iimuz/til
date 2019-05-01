# 50+ Data Structure and Algorithms Interview Questions for Programmers の挑戦

[50+ Data Structure and Algorithms Interview Questions for Programmers][hackernoon] の記事の問題を解いた時のメモです。

* [日本語訳][postd]

[hackernoon]: https://hackernoon.com/50-data-structure-and-algorithms-interview-questions-for-programmers-b4b1ac61f5b0
[postd]: https://postd.cc/50-data-structure-and-algorithms-interview-questions-for-programmers/

## テスト実行

各問題に対する解答と実際に動かすコードは別にしています。
動作確認はテストコードを用いて行うため、下記のようにして実行します。

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

## Table of Content

* problem001: 1 から 100 までの与えられた整数配列の中から足りない数字を探す。
* problem002: 与えられた整数の配列において重複した数字を探す。
* problem003: ソートされていない配列から最大値と最小値を探す。
  * ライブラリや組み込み関数を利用せずに実現する。
    python の場合は、 list に対して max や min が取得できる関数を使わない。
* problem004: 整数の配列と合計値が与えられたときに、整数の配列の中の 2 つの要素の組み合わせで、
  合計値と一致するパターンを全て探す。
* problem005: 整数の配列内に複数の重複がある場合に、重複を削除するにはどうすればよいか。
  * set などの Collection API を利用しない。

