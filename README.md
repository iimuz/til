# gcp-tiny-script

Tiny scripts for GCE

## アカウント切り替え

```sh
$ gcloud config configurations activate hoge
```

### 切り替え用アカウントの作成

```sh
$ gcloud config configurations create hoge
$ gcloud config configurations activate hoge
# 設定の作成
$ gcloud config set compute/region asia-northeast1
$ gcloud config set compute/zone asia-northeast-a
$ gcloud config set core/account hoge@example.com
$ gcloud config set core/project hoge-project
$ gcloud cofnig set core/disable_usage_reporting False
# 設定を作った後に認証が必要なため認証は実行しておく
$ gcloud auth login
```

