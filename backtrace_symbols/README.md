# The GNU Library の `backtrace` 関数を利用したバックトレースの取得

ここでは、 `backtrace` と `backtrac_symbols` を利用した方法を示します。

## 注意点

- `backtrace_symbols` 関数によって返る値は `free` する必要があります。
- 最適化オプションとして `-O2` などをつけると結果が変わります。
- `-rdynamic` オプションが必要です。

## Usage

実行には下記のようにする。

```sh
make build
make run
```

## Example

```txt
In main
In func1
In func2
===== trace result =====
../_bin/backtrace_symbols.out(+0x420b) [0x55ecf0f7220b]
../_bin/backtrace_symbols.out(+0x43fd) [0x55ecf0f723fd]
../_bin/backtrace_symbols.out(+0x4368) [0x55ecf0f72368]
../_bin/backtrace_symbols.out(main+0x3a) [0x55ecf0f72164]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7) [0x7f2ef4013b97]
../_bin/backtrace_symbols.out(_start+0x2a) [0x55ecf0f7204a]
==========
Out func2
Out func1
```