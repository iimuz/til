# Rust Programming Language 日本語版

[The Rust Programming Language 日本語版][rust-jp]に書かれた内容を環境構築含めて写経している。

- Chap.01
  - [hello_world](hello_world): `rustc main.rs` でバイナリを作成
  - [hello_cargo](hello_cargo)
    - `cargo new hello_cargo --bin`: バイナリ生成プロジェクトの新規作成
    - `cargo build`: デバッグ用バイナリの作成
    - `cargo build --release`: リリース用バイナリの作成
    - `cargo run`: デバッグ用バイナリを作成して即時実行
    - `cargo check`: バイナリを生成せずにビルドできるか確認する(cargo build より高速に実行できる)
- Chap.02
  - [guessing_game](guessing_game)
    - `Cargo.toml` の dependencies に rand クレートの依存を追加し build することで依存パッケージを自動でダウンロードする。
    - `cargo update`: パッケージ依存関係の更新
    - `cargo doc --open`: パッケージのドキュメントを生成して開く。ただしhtmlファイルが生成されるだけなので、サーバが起動するわけではない。

[rust-jp]: https://doc.rust-jp.rs/book-ja/title-page.html

## Tips

### Blocking waiting for file lock on package cache

下記のようなメッセージが発生してビルドできなくなることがある。

```txt
Blocking waiting for file lock on package cache
```

[ここ][zen-tanshio]にあるように `cargo clean` で解決できる場合もあるらしいが、 `~/.cargo/.package-cache` の消去で復旧できた。

[zen-tanshio]: https://zenn.dev/tanshio/articles/0cfbea0c2e2a29
