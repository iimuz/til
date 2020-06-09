# SQL Server Tips

## SQLCMD

コマンドから利用する場合は、 sqlcmd コマンドを利用します。
sqlcmd のコマンドは入力後に go と入れないと実行されません。

- ログイン: `sqlcmd -S server -U user -P password -d database`
  - パスワードは設定しなければ、パスワード入力用の何も表示されない入力ができます。
  - データベースは設定しなくても、ログイン後にセットできます。
- データベース一覧の取得: `select name from sys.databases;`
- データベースの作成: `create database name;`
- データベースの削除: `drop database name;`
- データベースの指定: `use name;`
- テーブル一覧の取得: `select name from sysobjects where xtype = 'U';`
- テーブルの作成:

  ```sql
  create table members(
    id int identity(1,1) primary key,
    name nvarchar(32),
    birthday datetime
  );
  ```

## Python 用のドライバ

Python から利用する場合は、 pyodbc を導入する必要があります。

```sh
pip install pyodbc
```

ただし、 pyodbc を ubuntu で利用する場合は、バイナリ配布ではなくビルドが必要となります。
ビルド時に下記パッケージがないとエラーとなります。

```sh
apt install unixodbc-dev
```

場合によっては、上記パッケージだけでは python からアクセスするときにドライバエラーが発生します。
その場合は、追加で下記を実施してください。

```sh
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo bash -c "curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list"
apt-get update
ACCEPT_EULA=Y apt-get install msodbcsql17
```

- 参考
  - [fatal error: sql.h: No such file or directory][mkleehammer]
  - [Microsoft ODBC Driver for SQL Server をインストールする (Linux)][ms_odbc_driver]

[mkleehammer]: https://github.com/mkleehammer/pyodbc/issues/441
[ms_odbc_driver]: https://docs.microsoft.com/ja-jp/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server
