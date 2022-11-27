# Stable diffusion を利用した画像生成

Google Colab 環境を利用して Stable Diffusion で画像を生成するノートブック。
実行環境は Google Colab の GPU ランタイムになります。
モデルの取得には Hugging Face の Access Tokens が必要になるため、Hugging Face のアカウントが必要になります。Hugging Face のアカウント取得後に `Settings -> Access Tokens` から Token を取得しノートブックでログインを実施してください。

- [`stable-diffusion-pipeline.ipynb`](stable-diffusion-pipeline.ipynb) [![Open in Colab][img00]](http://colab.research.google.com/github/iimuz/til/src/til-20220924/stable-diffusion-pipeline.ipynb): prompt から画像を生成する例。
- [`stable-diffusion-gradio-demo.ipynb`](stable-diffusion-gradio-demo.ipynb) [![Open in Colab][img00]](http://colab.research.google.com/github/iimuz/til/src/til-20220924/stable-diffusion-gradio-demo.ipynb): `stable-diffusion-pipeline.ipynb` と同様の内容に対して gradio を利用してデモ用の Web 画面を作成した例。
- [`20221022-stable-diffusion-img2img-pipeline.ipynb`](20221022-stable-diffusion-img2img-pipeline.ipynb) [![Open in Colab][img00]](http://colab.research.google.com/github/iimuz/til/src/til-20220924/20221022-stable-diffusion-img2img-pipeline.ipynb): img2img を利用して画像を生成する例。

[img00]: https://colab.research.google.com/assets/colab-badge.svg
