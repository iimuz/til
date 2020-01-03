# Jupyter Tips

[Jupyter][jupyter] での Tips 置き場です。

[jupyter]: https://jupyter.org/

## 画像ファイルの表示

画像ファイルを表示するだけならば下記のように記載できます。

```py
from IPython.display import display_png
from IPython.display import Image

display_png(Image("/path/to/png/file"))
```

- 参考文献
  - [Qiita: jupyter notebookで画像を表示させたいだけならば、`IPython.display`モジュールのメソッドを使うべし][knknkn1162]

[knknkn1162]: https://qiita.com/knknkn1162/items/77999450c59db915ab87

