# docker with GPU

GPU を含む docker 環境で Tensorflow や PyTorch が動作することをテストする簡単なスクリプトです。
現在(2020/2/3)の状況では docker-compose.yml から docker 19.03 でサポートされた GPU を動作させる方法がありません。
そのため、スクリプトで docker を起動します。
下記のテストを行うためのスクリプトを用意しています。

- docker 内で nvidia-smi により GPU のログを取得できる。
- Tensorflow で GPU が認識できているか確認する。
- MNIST のチュートリアルを実行する。

## docker での GPU 利用

docker で GPU を利用する場合は、 docker 19.03 以降であれば、 `--gpus all` フラグを追加します。
`--gpus all` フラグを利用する場合は、従来のように nvidia のイメージである必要はなく、
ノーマルのイメージのままでも GPU を利用できるようになります。
