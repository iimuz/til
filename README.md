# Python script for uploading to Google Photos

Google Photosへ画像をアップロードするためのスクリプトです。

## Development environments

* Ubuntu 18.04
* Python 3.7.3

## Usage

下記のようにコマンドを実行します。

```sh
$ python -m app
```

## Advance preparation

### Google Account and GCP Account

アップロード先はGoogle Photosとなります。
そのため、Googleアカウントが必要です。
また、認証情報を生成するためにGCPのプロジェクトが必要となります。

1. Googleアカウントの取得
1. [Google Developers Console][gcp_console]からGCPプロジェクトの作成
1. [Get Started with REST][photos_get-started]に従い、プロジェクトのGoogle Photos APIの有効化
1. OAuth2の認証用jsonを取得する
    * 認証情報 -> 認証情報を作成 -> OAuth クライアント ID
        * アプリケーションの種類は、何でもよいようです。
    * JSON形式でダウンロード

[gcp_console]: https://console.developers.google.com/
[photos_get-started]: https://developers.google.com/photos/library/guides/get-started
