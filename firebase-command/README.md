# firebase command

firebase の CLI を実行する環境を構築します。

## Usage

ログイン結果のファイルは `~/.config/configstore` に共通して保存し、使いまわすようにしています。
configstore の場所は、 uid が 1000 の場合は、下記のコマンドのように node ユーザのフォルダになります。
それ以外の場合は、おそらくルート (`/.config/configstore`) になるはずです。

```sh
$ docker build --force-rm -t localhost:firebase .
$ mkdir -p ~/.config/configstore
$ docker run -it -v ~/.config/configstore:/home/node/.config/configstore:rw -p 9005:9005 -u $(id -u):$(id -g) localhost:firebase firebase login

i  Firebase optionally collects CLI usage and error reporting information to help improve our products. Data is collected in accordance with Google's privacy policy (https://policies.google.com/privacy) and is not used to identify you.

? Allow Firebase to collect CLI usage and error reporting information? Yes
i  To change your data collection preference at any time, run `firebase logout` and log in again.

Visit this URL on this device to log in:
https://accounts.google.com/o/oauth2/auth?xxxxxxxxxxxxxxxxxxx

Waiting for authentication...

✔  Success! Logged in as hoge@example.com

$ docker run -it --rm -v ~/.config/configstore:/home/node/.config/configstore:rw -p 9005:9005 -u $(id -u):$(id -g) localhost:firebase firebase projects:list

✔ Preparing the list of your Firebase projects
┌──────────────────────┬────────────┬──────────────────────┐
│ Project Display Name │ Project ID │ Resource Location ID │
├──────────────────────┼────────────┼──────────────────────┤
│ hogehoge             │ hogehoge   │ [Not specified]      │
└──────────────────────┴────────────┴──────────────────────┘

1 project(s) total.
```

## 参考資料

- 2018.12.11: [Firebase Cloud Functionsで消耗したくない人のために、開発環境のベストプラクティスをまとめていったらDockerに行き着いた話（随時更新）][pannpers]

[pannpers]: https://qiita.com/pannpers/items/244a7e3c18d8c8422e4f

