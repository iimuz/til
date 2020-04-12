# Jupyter Tips

[Jupyter][jupyter] での Tips 置き場です。
実行環境として [Google Colaboratory][colab] を利用した場合の Tips も併記しています。

[colab]: https://colab.research.google.com/
[jupyter]: https://jupyter.org/

## モジュールのリロード

jupyter では一度読み込んだモジュールは、カーネル実行中に他の場所で書き換えられたとしても、
自動的には再読み込みしません。
ただし、 [IPython extensions autoreload][autoreload] にあるように、
マジックコマンドを利用してモジュールの再読み込みが可能となります。

```py
%load_ext autoreload
%autoreload 2
```

- `%autoreload 0`: リロードなし
- `%autoreload 1`: `%aimport` で指定したモジュールは実行前にリロード
- `%autoreload 2`: 全モジュールをリロード

上記以外にも、 `%aimport` を利用することで特定のモジュールを読み込み対象として指定したり、
除外したりといったことが可能です。

[autoreload]: https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html

## seaborn の設定

[seaborn][sns] を利用すると [matplotlib][matplotlib] の表示が綺麗になります。
通常通り matplotlib を利用して描画した際に seaborn の表示を利用するためには、
予め下記のコマンドを実行しておく必要があります。

```py
import seaboard as sns

sns.set()
```

- [seborn.set][sns_set]

[matplotlib]: https://matplotlib.org/
[sns]: https://seaborn.pydata.org/index.html
[sns_set]: https://seaborn.pydata.org/generated/seaborn.set.html

## 画像ファイルの表示

画像ファイルを表示するだけならば下記のように記載できます。

```py
from IPython.display import display_png
from IPython.display import Image

display_png(Image("/path/to/png/file"))
```

参考文献

- [Qiita: jupyter notebook で画像を表示させたいだけならば、`IPython.display`モジュールのメソッドを使うべし][knknkn1162]

[knknkn1162]: https://qiita.com/knknkn1162/items/77999450c59db915ab87

## (Colab) Google drive のマウント

Google Colaboratory では Google Drive をマウントして利用することができます。

```py
import google.colab.drive as drive

drive.mount("path/to/mount/dir")
```

実行後に URL が表示されるので、リンクをクリックして、 Google Account で許可します。
その後、表示されるパスワードをコピーして入力することで、
自分のアカウントの Google Drive がマウントされます。
注意点としては、実際に自分の Google Drive のファイルが置かれている場所は、
`path/to/mount/dir/My Drive` になることです。

## (Colab) Kaggle API

Colaboratory から [Kaggle][kaggle] を利用する際には、 [Kaggle API][kaggel_api] が利用できます。

```py
!pip install kaggle
!kaggle competitions list
```

ただし、 Kaggle API を利用するためには、 Token が必要となります。
毎回 Colaboratory を起動するたびに取得しなおすのは大変なので、
Google Drive からコピーして配置すると楽にできます。
この時、 Colaboratory 上で `kaggle.json` を配置する先は、下記になります。

- `/root/.kaggle/kaggle.json`

また、アクセス権を `chmod 600 /root/.kaggle/kaggle.json` としておく必要があります。

[kaggle]: https://www.kaggle.com/
[kaggel_api]: https://github.com/Kaggle/kaggle-api

### Kaggle API によるデータのダウンロード

Kaggle API を利用してデータセットをダウンロードするときは、コンペ名称から指定すると楽です。
また、何もオプションを設定しないと、直下にダウンロードするためオプションで配置先が指定できます。

```py
!kaggle competitions download -c demand-forecasting-kernels-only -p path/to/raw
```

## PyTorch のシード固定

pytorch を利用する際にランダムシードを固定する処理です。

```py
import random

import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
```
