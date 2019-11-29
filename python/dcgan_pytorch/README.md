# DCGAN using PyTorch

Implemtation of Deep Convolutional Generative Adversarial Networks (DCGAN) using PyTorch.

PyTorch を利用して DCGAN を実装してみるサンプルになります。

## Usage

```sh
python train.py \
  --batch_size 127 \
  --learning_rate 2e-4 \
  --epochs 25 \
  --z_dim 62 \
  --checkpoint_images 64 \
  --log_dir "logs" \
  --cuda
```

実際に Google Colaboratory を利用して実行した際の ipynb は、
notebooks フォルダの下にあります。

## References

* [arXiv: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks][arxiv]
* [人工知能に関する断創録: PyTorch (12) Generative Adversarial Networks (MNIST)][aidiary]
* [Re:ゼロから始めるML生活: 【論文メモ:DCGAN】Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks][tsunotsuno]
* [Qiita: GANについて概念から実装まで　～DCGANによるキルミーベイベー生成～][taku]
* [Qiita: pytorchで書いたDCGANでアニメキャラの顔を生成する][phyblas]
* [Qiita: ５ステップでできるPyTorch - DCGAN][hokuto]

[aidiary]: http://aidiary.hatenablog.com/entry/20180304/1520172429
[arxiv]: https://arxiv.org/abs/1511.06434
[hokuto]: https://qiita.com/hokuto_HIRANO/items/7381095aaee668513487
[phyblas]: https://qiita.com/phyblas/items/bcab394e1387f15f66ee
[taku]: https://qiita.com/taku-buntu/items/0093a68bfae0b0ff879d
[tsunotsuno]: https://tsunotsuno.hatenablog.com/entry/dcgan

