---
title: AWS Hands-on for Begginers Serverless
date: 2022-12-10
lastmod: 2022-12-17
---

下記の 2 つを実行したときのメモ。
1 つ目は、手動でリソースを作成する。2 つ目で SAM を利用して実施する方法を説明している。

- [AWS Hands-on for Begginers Serverless #1](https://pages.awscloud.com/event_JAPAN_Hands-on-for-Beginners-Serverless-2019_LP.html)
- [AWS Hands-on for Beginners Serverless #2](https://pages.awscloud.com/event_JAPAN_Ondemand_Hands-on-for-Beginners-Serverless-2_LP.html)

## 実行方法

### sam-app-delete

下記の手順で sam コマンドを利用することでデプロイすることができる。また、デプロイしたリソースの削除を実行可能である。

```sh
# ローカル環境で作成して実行できるか確認
sam build --use-container
sam local invoke --container-host host.docker.internal

# デプロイと削除の実行
sam deploy --profile hoge
sam delete --profile hoge
```

## sam-app-delete に関するメモ

`sam-app-delete` フォルダ: [AWS SAM の delete コマンドを試す](https://www.d-make.co.jp/blog/2021/08/21/try-aws-sam-delete-command/)を参考に `sam init` から `sam delete` までを試したフォルダ。

### sam-app-delete の作成手順

sam-app-delete は下記の手順で作成している。

```sh
$ sam init
# 1 - AWS Quick Start Templates
# 1 - Hello World Example
# Use the most popular runtime and package type? (Python and zip): y
# Would you like to enable X-Ray tracing on the function(s) in your application?: N
# Project name: sam-app-delete
$ cd sam-app-delete
$ sam build --use-container
$ sam deploy --guided --profile hoge
# Stack name: sam-app-delete
# AWS Region: ap-northeast-1
# Confirm changes before deploy: y
# Allow SAM CLI IAM role creation: y
# Disable rollback: N
# HelloWorldFunction may not have authorization defined, Is this okay?: y
# Save arguments to configuration file: y
# SAM configuration file: samconfig.toml
# SAM configuration environmen: default
# Deploy this changeset?: N
```

### container 内から sam コマンドを実行する場合

docker 内に sam を導入している場合は、 `sam local invoke` 実行時に `--container-host host.docker.internal` オプションが必要。

```sh
sam local invoke --container-host host.docker.internal
```
