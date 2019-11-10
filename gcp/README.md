# Today I Learn about GCP

- firebase-command: firebase CLI を docker
- firebase-static-site: firebase hosting を利用したサンプル
- firebase-with-cloud-run: firebase hosting と cloud run の連携
- scripts: GCP を操作するための簡易スクリプト

## Tips

### アカウント切り替え

```sh
$ gcloud config configurations activate hoge
```

#### 切り替え用アカウントの作成

```sh
$ gcloud config configurations create hoge
$ gcloud config configurations activate hoge
# 設定の作成
$ gcloud config set compute/region asia-northeast1
$ gcloud config set compute/zone asia-northeast-a
$ gcloud config set core/account hoge@example.com
$ gcloud config set core/project hoge-project
$ gcloud config set core/disable_usage_reporting False
# 設定を作った後に認証が必要なため認証は実行しておく
$ gcloud auth login
```

### compute instance のイメージ名一覧

```
$ gcloud compute images list
```

