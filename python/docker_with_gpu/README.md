# docker with GPU

GPU を含む docker 環境で Tensorflow や PyTorch が動作することをテストする簡単なスクリプトです。
現在(2020/2/3)の状況では docker-compose.yml から docker 19.03 でサポートされた GPU を動作させる方法がありません。
そのため、スクリプトで docker を起動します。
下記のテストを行うためのスクリプトを用意しています。

- `nvidia_smi.sh`: docker 内で nvidia-smi により GPU のログを取得できる。
- `check_gpu_using_tf.sh`: Tensorflow で GPU が認識できているか確認する。
  - `Num GPUs Available: 1` のようなログが出ます。数字は利用できる GPU 数に依存します。
- `logging_device_placement.sh`: GPU で簡単な計算を行い、計算したデバイスを表示する。
  - `Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0` のようなログが出力されます。
- `mnist_tutorial_tf.sh`: MNIST のチュートリアルを実行する。

## docker での GPU 利用

docker で GPU を利用する場合は、 docker 19.03 以降であれば、 `--gpus all` フラグを追加します。
`--gpus all` フラグを利用する場合は、従来のように nvidia のイメージである必要はなく、
ノーマルのイメージのままでも GPU を利用できるようになります。

## Tensorflow の docker イメージ

Tensorflow は v2 系の場合、インストールするパッケージを CPU と GPU で分ける必要はなくなっています。
しかしながら、 docker イメージは、 gpu のタグが入ったイメージを利用する必要があります。

- `tensorflow/tensorflow:2.1.0-gpu-py3`: GPU が利用できるイメージ
- `tensorflow/tensorflow:2.1.0-py3`: nvidia-smi は使えるが、 Tensorflow が GPU を認識しない。
