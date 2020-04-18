# PyTorch Tips

[PyTorch][pytorch] 関連のチップス集です。

[pytorch]: https://pytorch.org/

## PyTorch の再現性確保

PyTorch でランダム値が利用されるために毎回異なる結果となる部分を固定するための方法です。
PyTorch を利用する際にランダムシードを固定する処理です。
GPU の場合は完全に一致させる部分は利用するとかなり遅くなります。
ある程度一緒であればよいのであれば、利用しないほうが良いです。

```py
import random

import numpy as np
import torch

def worker_init_fn(worker_id):
    random.seed(worker_id)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu の場合に必要
cudnn.deterministic = True  # gpu の場合は必要だが遅くなる。

test_MNIST = datasets.MNIST("./data", train=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_MNIST,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn
)
```

参考資料

- 2018.8.13 Qiita [pytorch で CNN の loss が毎回変わる問題の対処法 (on cpu)][chat-flip]

[chat-flip]: https://qiita.com/chat-flip/items/4c0b71a7c0f5f6ae437f

## python multi processing における vscode でのデバッグ

pytorch の dataloader の worker 数を複数にしていると vscode でデバッグしようとすると、
下記のようなエラーが発生します。

```txt
Exception escaped from start_client
failed to launch debugger for child process
AssertionError: can only join a child process
RuntimeError: already started
```

下記の点を追加するとできるようです。

```py
import multiprocessing

if __name___ == "__main__":
    multiprocessing.set_start_method("spawn", True)
```

加えて、 vscode の launch.json に下記を追加します。

```json
{
  "version": "0.1.0",
  "configurations": [
    {
      "name": "Python debug",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "subProcess": true, // new
      "console": "integratedTerminal"
    }
  ]
}
```

参考資料

- 2020.4.12 Qiita [python のマルチプロセスプログラムを VSCode でデバッグする][stat]

[stat]: https://qiita.com/stat/items/544e7286e5e5ffa4763e
