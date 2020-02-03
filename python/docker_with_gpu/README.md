# docker with GPU

GPU を含む docker 環境で Tensorflow や PyTorch が動作することをテストする簡単なスクリプトです。
現在(2020/2/3)の状況では docker-compose.yml から docker 19.03 でサポートされた GPU を動作させる方法がありません。
そのため、スクリプトで docker を起動します。
下記のテストを行うためのスクリプトを用意しています。

- `nvidia_smi.sh`: docker 内で nvidia-smi により GPU のログを取得できる。
- tensorflow 系
  - `tf_check_gpu.sh`: Tensorflow で GPU が認識できているか確認する。
    - `Num GPUs Available: 1` のようなログが出ます。数字は利用できる GPU 数に依存します。
  - `tf_logging_device_placement.sh`: GPU で簡単な計算を行い、計算したデバイスを表示する。
    - `Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0` のようなログが出力されます。
  - `tf_mnist_tutorial.sh`: MNIST のチュートリアルを実行する。
- pytorch 系

  - `torch_check_gpu.sh`: GPU の認識状態を確認します。

    ```sh
    current device: 0
    device: <torch.cuda.device object at 0x7f7972427050>
    device count: 1
    device name: Tesla M60
    available: True
    ```

## docker での GPU 利用

docker で GPU を利用する場合は、 docker 19.03 以降であれば、 `--gpus all` フラグを追加します。
`--gpus all` フラグを利用する場合は、従来のように nvidia のイメージである必要はなく、
ノーマルのイメージのままでも GPU を利用できるようになります。

## Tensorflow の docker イメージ

Tensorflow は v2 系の場合、インストールするパッケージを CPU と GPU で分ける必要はなくなっています。
しかしながら、 docker イメージは、 gpu のタグが入ったイメージを利用する必要があります。

- `tensorflow/tensorflow:2.1.0-gpu-py3`: GPU が利用できるイメージ
- `tensorflow/tensorflow:2.1.0-py3`: nvidia-smi は使えるが、 Tensorflow が GPU を認識しない。

## 参考資料

- 2018.1.8 StackOverflow [How to check if pytorch is using the GPU?][stackoverflow_torch_gpu]
  - PyTorch を利用した GPU デバイスの認識状態を取得するためのコードが記載されています。
- PyTorch
  - [TRAINING A CLASSIFIER][torch_cifar10]
    - PyTorch の CIFAR10 の学習スクリプトの元です。
      CPU 番のため `.cuda()` をつけて GPU 版にしています。

[stackoverflow_torch_gpu]: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
[torch_cifar10]: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
