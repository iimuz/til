# nullptr の実体を参照渡しした時の挙動確認

nullptr の実体へ変換して関数に渡すことができることを確認する。
一方で、 nullptr の実体へ返還したデータは、メンバ関数を呼び出すとセグメンテーション違反となる。

## Usage

```sh
make build
make run
```
