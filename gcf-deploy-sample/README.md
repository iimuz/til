# GCF へのデプロイサンプル

Google Cloud Functions (GCF) へ master へのコミットから自動でデプロイする環境構築を試しています。
今回の構成は下記のようになっています。

1. 開発者が GitHub (master) へ push する
1. TravisCI が GitHub (master) へのコミットをもとに Cloud Source Repositories へ push する
1. TravisCI が Cloud Source Repository の内容を元に GCF を更新する

TravisCI を経由している理由は、もしビルドが必要な部分があれば、
TravisCI でテストやビルドを行って、
その成果物を Cloud Source Repositories へ push することを想定したためです。

TravisCI から Cloud Source Repositories へ push するためには、
サービスアカウントに Source Repository への書き込み権限が必要になるはずです。

また、 TravisCI から Cloud Functions へデプロイするためには、
Cloud Functions 開発者の権限に加えて、
サービスアカウントユーザという役割をサービスアカウントに付与する必要がありました。
