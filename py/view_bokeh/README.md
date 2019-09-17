# Bokeh Application with Server

Bokeh の Web Application を作成する機能のサンプルです。
Bokeh のサーバ機能を用いることでブラウザからアクセスすることが可能です。

下記資料を参考にしています。

- [Qiita: Bokehが素晴らしすぎてわずか130行で対話的可視化ツールを作ってしまった話][kimisyo]

[kimisyo]: https://qiita.com/kimisyo/items/8aac9c5a08d883d94bbb

## 実行コマンド

```sh
bokeh serve view_bokeh/server.py
```

bokeh コマンドは、 pip で bokeh を導入した時に利用可能となります。
また、 serve コマンドのオプションなどは、 [公式ドキュメント][serve] を参照してください。

[serve]: https://bokeh.pydata.org/en/latest/docs/reference/command/subcommands/serve.html

## Tips

- csv にヘッダが付与されていると読み込みに失敗する。

### `if __name__ == '__main__':`

実行コマンドと考えて、 main 関数を `if __name__ == '__main__':` 以下に記述すると正常に動作しない。

### テスト用データ作成

テスト用に iris.csv を生成するスクリプトを保存します。
実行方法は下記のようになります。

```sh
cd ..
python view_bokeh/get_irirs_dataset
```

### ログ出力

ログ情報を出力する場合は、 bokeh の実行時にログレベルを設定する必要があります。

```sh
bokeh serve server.py --log-level info
```

## 要求機能

現時点で触ってみて足りないと感じる機能を記述する。

- feat: csv ファイルのアップロード機能
- feat: 表示領域に合わせた配置
  - 表示領域が小さくてもスクロールバーで対応されてしまう。
- feat: hover または data 選択によって画像表示を行う機能。
- bug: iris データで hover を表示すると、 `???` というのが表示される。

## 参考資料

- [Bokeh][bokeh]

[bokeh]: https://bokeh.pydata.org/en/latest/
