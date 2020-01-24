# Bash Tips

## ファイル数カウント

ファイル数をカウントする方法です。
find を利用しているためディレクトリ以下を再帰的に探索します。

```sh
find /path/to/dir -type f | wc -l
```

参考資料

- 2013.10.2 Qiita [ディレクトリ内のファイル数をカウントする][stc1988]

[stc1988]: https://qiita.com/stc1988/items/e3a1d7dccafe4ab573fa

