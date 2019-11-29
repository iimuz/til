# nginx-example

Nginxを簡易に利用したい場合のdockerサンプル

## Usage

前提条件としてdockerが必要となります。

下記コマンドで起動できます。

```bash
$ docker-compose up -d
```

接続するには下記URLへアクセスしてください。

```
$ curl http://localhost:8080/
```

終了するときは、下記コマンドで終了します。

```bash
$ docker-compose down -v
```
