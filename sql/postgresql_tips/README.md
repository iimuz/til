# PostgreSQL Tips

## コマンドオプション

`psql` を利用する時のコマンドオプションで良く利用するものです。

- `-h localhost`: DB のホストを指定します。
- `-p 5432`: DB のポートを指定します。
- `-U postgres`: ユーザを指定します。
- `-d database_name`: データベース名を指定します。
- `-f hoge.sql`: スクリプトファイルを入力として実行します。
- `-o hoge.txt`: 出力を指定したファイルに書き出します。
- `PGPASSWORD=hogeoge psql`: パスワードを指定して起動します。
  - 環境変数の `PGPASSWORD` をパスワードとして読みに行くため、
    別の方法で指定しても問題ありません。

## テーブルのデータ容量確認

- 2019.5.17 Qiita [PostgreSQL で各テーブルの総サイズと平均サイズを知る][awakia]
  - 簡易バージョン: `table_size.sql`
  - pg_toast 考慮バージョン: `table_size_sp.sql`

[awakia]: https://qiita.com/awakia/items/99c3d114aa16099e825d
