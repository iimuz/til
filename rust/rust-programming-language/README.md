# Rust Programming Language 日本語版

[The Rust Programming Language 日本語版][rust-jp]に書かれた内容を環境構築含めて写経している。

[rust-jp]: https://doc.rust-jp.rs/book-ja/title-page.html

## Chap.01

- [hello_world](hello_world): `rustc main.rs` でバイナリを作成
- [hello_cargo](hello_cargo)
  - `cargo new hello_cargo --bin`: バイナリ生成プロジェクトの新規作成
  - `cargo build`: デバッグ用バイナリの作成
  - `cargo build --release`: リリース用バイナリの作成
  - `cargo run`: デバッグ用バイナリを作成して即時実行
  - `cargo check`: バイナリを生成せずにビルドできるか確認する(cargo build より高速に実行できる)

## Chap.02

- [guessing_game](guessing_game)
  - `Cargo.toml` の dependencies に rand クレートの依存を追加し build することで依存パッケージを自動でダウンロードする。
  - `cargo update`: パッケージ依存関係の更新
  - `cargo doc --open`: パッケージのドキュメントを生成して開く。ただし html ファイルが生成されるだけなので、サーバが起動するわけではない。

## Chap.03

- [variables](variables)
  - `const`を利用する場合は、値の方を必ず付ける必要がある。
  - 定数は定数式にしか設定できず、関数呼び出し結果や実行時に評価される値には設定できない。
  - 定数の命名規則は大文字でアンダースコア区切り。(`MAX_POINTS`)
  - シャドーイングは変数を下辺にするのとは異なり、let キーワードを使わないとコンパイルエラーとなる。
  - シャドーイングの場合は、実行的には新しい変数を作っていることになるので型の異なる同一の変数名を使いまわせる。
  - rust における数値の基準型は `i32` と `f64` となる。
  - tuple 型のアクセスは`x.0`, `x.1`のようになる。
    一方で、配列型のアクセスは`x[0]`, `x[1]`になる。
  - 配列外へのアクセスは読み込みであっても実行時エラーとなる。(C だと読み込みはエラーしないが、Rust ではできない。)
- [functions](functions)

  - Rust における関数の命名規則は、スネークケースとなる(`some_variable`)。
  - '文'とは何らかの動作をして値を返さない命令。
    '式'とは結果値に評価される命令。

    ```rs
    fn main() {
        let x = 6;  // 文

        let y = {
            let x = 3;  // 文
            x + 1  // 式(セミコロンがつかない)
        };  // 文
    }
    ```

  - rust では `let` は文となるため、 `x = y = 6` のような記述はできない。
  - 関数ブルおっくからの最後の式が戻り値となる。return で早期に返すことも可能。

- [branches](branches)
  - rust では論理値以外が自動で論理値に変換されないため if 文で 0 とそれ以外の整数のような条件設定はできない。
  - `if` は式である。
- [loops](loops)
  - 配列に対しては `for element in a.iter()` で要素のループを実行できる。
  - `Range` 型として 1, 2, 3 を表す時は `(1..4)` と書ける。
  - `rev()` によって逆順にすることができる。

## Chap.04

- スタックとヒープ
  - スタックは last in, first out であり、高速に動作する。
    スタック上のデータはすべて既知の固定サイズである必要がある。
  - ヒープはメモリを確保(allocate)する必要がある。
    スタックよりもデータアクセスは低速である。
- スコープを抜けるときに確保したメモリを解放する`drop`が自動で呼ばれる。
- 変数に束縛する際にスタック上のデータはコピーされるが、ヒープ上のデータはポインタがコピーされる。

  ```rs
  let x = 5;
  let y = 5;  // 2つの変数(x, y)が存在し、両方の値は5になる。

  let s1 = String::from("hello");
  let s2 = s1;  // s1とs2は同一のヒープ上の"hello"を表す。shallowコピーではなくmoveとなりs1は無効化される。
  ```

- 自動的にヒープデータの deep copy は実行されない。
  ヒープデータの deep copy が必要な場合は、 `clone` メソッドを呼ぶ。

  ```rs
  let s1 = String::from("hello");
  let s2 = s1.clone();
  ```

- Copy トレイトに適合すると整数値のように代入後も古い変数が利用可能となる。
  Drop トレイトを実装している場合は、Copy トレイトによる注釈はできない。
- `&` を付与することで参照渡しが可能となる。
- `&mut` を付与することで可変な参照を利用できる。
- 特定スコープで特定データに対しては一つしか可変な参照を持つことはできない。
- 参照の規則
  - 一つの可変参照もしくは不変な参照を複数のいずれかを利用できる。両方は使えない。
  - 参照は常に有効でなければならない。
- `&s[0..2]`: スライスを利用できる。
  - `&s[..2]`: 最初が 0 開始なら省略可能。
  - `&s[2..]`: 末尾までであれば最後を書かなければよい。
  - `&s[..]`: 何も書かなければ全体を表す。
- 文字列スライスを渡すように構成する方が良い。

  ```rs
  fn first_word(s: &String) -> &str {} // かけるが良くない。
  fn first_word(s: &str) -> &str {} // より汎用性のある書き方。
  ```

- 文字列スライス以外は、例えば `&[i32]` のような型となる。

## Tips

### Blocking waiting for file lock on package cache

下記のようなメッセージが発生してビルドできなくなることがある。

```txt
Blocking waiting for file lock on package cache
```

[ここ][zen-tanshio]にあるように `cargo clean` で解決できる場合もあるらしいが、 `~/.cargo/.package-cache` の消去で復旧できた。

[zen-tanshio]: https://zenn.dev/tanshio/articles/0cfbea0c2e2a29
