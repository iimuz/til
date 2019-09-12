# firebase hosting with cloud run

firebase と cloud run の組み合わせを行うサンプルです。

## 手順

- docker image のビルド: `gcloud builds submit --tag gcr.io/projectID/helloworld`
- cloud run: `gcloud beta run deploy --image gcr.io/projectID/helloworld`

## Tips

- 下記のサービス API を有効化する必要がある。
  - Cloud Build
  - Cloud Run
  - Container Registory

## 参考資料

- [Cloud Run を使用した動的コンテンツの配信とマイクロサービスのホスティング][official]

[official]: https://firebase.google.com/docs/hosting/cloud-run

