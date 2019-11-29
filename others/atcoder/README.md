# atcoder

[AtCoder][atcoder] に挑戦した記録です。

[atcoder]: https://atcoder.jp/

## c++ の実行方法

基本的には、各コンテストの docker 環境と Makefile を利用します。
例えば、 ABC121 を例にすると下記のような手順となります。

```sh
$ cd ABC121/tools
$ docker-compose run --rm dev
# docker コンテナ内に入った状態となる
$ cd A
$ make build-gtest # テスト用に gtest をビルド(初回のみ必要)
$ make build-test
$ make run-test
$ cd ../B
$ make build-test
$ make run-test
```

