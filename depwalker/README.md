# Analyzer for Dependency Walker

[Dependency Walker][dependencywalker] でエクスポートしたファイルを解析する。

[dependencywalker]: http://www.dependencywalker.com/

## 実行コマンド

```sh
bokeh serve depwalker/link.py
```

bokeh コマンドは、 pip で bokeh を導入した時に利用可能となります。
また、 serve コマンドのオプションなどは、 [公式ドキュメント][serve] を参照してください。

[serve]: https://bokeh.pydata.org/en/latest/docs/reference/command/subcommands/serve.html

## 実行サンプル

vscode 1.36.1 の code.exe を dependency walker で出力したファイルの表示結果。

| 起動時の表示  | 特定のモジュールに限定した時の表示 |
| :----------: | :--------------------------------: |
| ![][sample]  |          ![][sample_dep]           |

[sample]: docs/code_sample.jpg
[sample_dep]: docs/code_sample_dep.jpg
