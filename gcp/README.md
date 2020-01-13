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
$ gcloud config set compute/zone asia-northeast1-a
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

### GCE の home だけ別のディスクをマウント

home 下に別のディスクをマウントする方法です。
想定としては、 Google Cloud Shell の自分用のような感じです。
基本的に使わないときは、 VM を削除して節約し、使うときにだけ VM を必要なインスタンスで生成します。
Google Cloud Shell だとホームディレクトリに 5GB しかなく、
ルート下 6GB 程度しかないため、 docker を使うと簡単に容量が足りなくなります。
それの代替手段として利用することを想定しています。

```sh
# home directory にするディスクの作成
# 必要ならばあとで大きくすればいいので、とりあえずのサイズで作る。
DISK_NAME=hoge-disk
gcloud compute disks create $DISK_NAME \
  --size 10 \
  --type pd-standard

# 作成したディスクはフォーマットが必要なため、ディスクをマウントした VM 上で下記を実行します。
INSTANCE_NAME=dev
MACHINE_TYPE=n1-standard-1
DISK_SIZE=30GB
gcloud compute \
  instances create $INSTANCE_NAME \
  --machine-type=$MACHINE_TYPE \
  --subnet=default \
  --network-tier=PREMIUM \
  --no-restart-on-failure \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --image=ubuntu-1804-bionic-v20200108 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-standard \
  --boot-disk-device-name=$INSTANCE_NAME \
  --disk=name=$DISK_NAME,device-name=$DISK_NAME,mode=rw,boot=no

sudo lsblk  # ディスクのデバイス ID 確認
DEVICE_ID=sdb
DEVICE_UUID=`sudo blkid -s UUID -o value /dev/$DEVICE_ID`
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/$DEVICE_ID

# 初回のみディスクの内容を HOME からコピーします。
sudo mkdir -p /mnt/disks/$DISK_NAME
sudo mount -o discard,defaults /dev/$DEVICE_ID /mnt/disks/$DISK_NAME
sudo rsync -av /home/ /mnt/disks/$DISK_NAME

# ディスクを HOME としてマウントするように設定して再起動します。
sudo rm -rf /home/*
echo UUID=$DEVICE_UUID /home ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
sudo reboot
```

