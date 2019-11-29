# Python script for uploading to Google Photos

Google Photosへ画像をアップロードするためのスクリプトです。

## Development environments

* Ubuntu 18.04
* Python 3.7.3

## Usage

下記のようにコマンドを実行します。

```sh
# 初回のみ認証設定
$ python -m auth login
# hoge.png画像をアップロード
$ python -m image upload hoge.png
# アルバム情報の確認
$ python -m album list
```

その他のコマンドに関しては、下記のようにそれぞれでヘルプを見れます。

```sh
$ python -m auth --help
$ python -m image --help
$ python -m album --help
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
