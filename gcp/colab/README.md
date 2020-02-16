# Colaboratory with ssh

Google Colaboratory に ssh で接続する方法です。
ただし、試した環境ではそれなりに動作がもたつきます。
編集方法などは、考えたほうがいいです。
VSCode の remote development で接続して編集すれば、編集自体は少しだけ楽に出来ました。
vim などだと、遅延が非常に分かりやすく編集が難しかったです。

## 設定

Colaboratory で下記を実行し、 sshd を起動します。
ここで設定したパスワードが ssh で接続するときのパスワードになります。

```py
import random
import string

password = ''.join(
    random.choice(string.ascii_letters + string.digits)
    for _ in range(20)
)
print(password)

!apt-get install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen
!echo root:$password | chpasswd
!mkdir -p /var/run/sshd
!echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
!echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
!echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
!echo "GatewayPorts yes" >> /etc/ssh/sshd_config
!echo "export LD_LIBRARY_PATH=/usr/lib64-nvidia" >> /root/.bashr

get_ipython().system_raw("/usr/sbin/sshd -D &"
```

その後、 Colaboratory 側から ssh で接続したい端末に向けてポートフォワーディングを行います。

```py
import subprocess

proc = subprocess.run(["setsid", "ssh", "user@ip", "--", "-fNR 20022:localhost:22", "-oStrictHostKeyChecking=no"])
```

## 永続化

データの退避先として Google Drive も利用可能です。
しかしながら、アクセス権などが通常と異なる状態になるため、 git 管理などが難しいです。
そこで、下記のスクリプトにより別のサーバにファイルを同期します。

- `sync_remote.sh`: 最初に Colaboratory にサーバからデータを移動し、開発環境を整えるスクリプトです。
- `sync_local.sh`: 定期的に監視しているフォルダ以下のファイルに更新があった場合にサーバにデータを転送します。

`sync_remote.sh` はサーバ側で実行します。
また、 `sync_local.sh` は Colaboratory で実行します。
実行には subprocess モジュールを用いて下記のように行います。

```py
import subprocess

proc = subprocess.Popen(["setsid", "bash", "sync_local.sh"]
```

## Colaboratory から接続するサーバ

Colaboratory から接続するサーバは、公開されたアドレスが必要となります。
自前で用意できない場合は、 GCE で用意し、その external ip に接続します。
その後、 自分のクライアント端末から GCE のサーバを踏み台として Colaboratory に接続します。

GCE サーバに Colaboratory から接続する場合は、下記のコマンドを実行すると接続できます。
途中で key の設定など聞かれますが、全て Enter で動作します。

```py
import subprocess

!gcloud config set compute/region hoge
!gcloud config set compute/zone hoge
!gcloud config set core/account hoge@example.cm
!gcloud config set core/disable_usage_reporting False
!gcloud auth login
!gcloud config set core/project hoge_project
!gcloud compute config-ss

proc = subprocess.run(["setsid", "gcloud", "compute", "ssh", "user@instance", "--", "-fNR 20022:localhost:22", "-oStrictHostKeyChecking=no"]
```
