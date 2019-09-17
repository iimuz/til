# ファイルコピーの処理時間比較

C++ においてファイルコピーにかかる時間を比較します。

## 比較方法

- streambuf_iterator
  - 1 byte 単位での読み込みと書き込みを行うため速度が遅い可能性がある。
  - 遅いという指摘もあったが、実験ではそれほど有意な差は見られなかった。
- rdbuf
  - ファイル内容を一括でメモリに読み込むため、巨大なファイルではメモリ枯渇の危険性がある。
- FILE
  - バッファサイズを任意で決められる。
  - stream に対して高速と指摘されている文献もあるが、有意な差は見られなかった。
- fstream
  - バッファサイズを任意で決められる。

## Usage

実行には下記のようにする。

```sh
make build
make run
```

## Example

```txt
=== stream buf ===
min: 10 [msec], max: 160 [msec], mean: 79.1 [msec]
=== rdbuf ===
min: 47 [msec], max: 232 [msec], mean: 106.9 [msec]
=== fstream[1kB] ===
min: 47 [msec], max: 135 [msec], mean: 76.3 [msec]
=== fstream[256kB] ===
min: 32 [msec], max: 278 [msec], mean: 85.7 [msec]
=== file[1kB] ===
min: 36 [msec], max: 216 [msec], mean: 79.1 [msec]
=== file[256kB] ===
min: 31 [msec], max: 177 [msec], mean: 92.8 [msec]
```