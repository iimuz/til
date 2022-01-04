# Rust Programming Language 日本語版

[The Rust Programming Language 日本語版][rust-jp]に書かれた内容を環境構築含めて写経している。

- Chap.01
  - [hello_world][hello_world]: `rustc main.rs` でバイナリを作成
  - [hello_cargo][hello_cargo]
    - `cargo new hello_cargo --bin`: バイナリ生成プロジェクトの新規作成
    - `cargo build`: デバッグ用バイナリの作成
    - `cargo build --release`: リリース用バイナリの作成
    - `cargo run`: デバッグ用バイナリを作成して即時実行
    - `cargo check`: バイナリを生成せずにビルドできるか確認する。(cargo buildより高速に実行できる)

[rust-jp]: https://doc.rust-jp.rs/book-ja/title-page.html
