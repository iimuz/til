# Python Tips

## 親ディレクトリ経由の import

下記のようなディレクトリ構成において `module1.py` から `module2.py` を呼び出す方法です。
普通に import を書いた場合、 `cd root; python package1/module1.py` とすると、
`module2.py` が見つからないというエラーが発生します。

```txt
root
|-package1
|  |- module1.py
|-package2
   |- module2.py
```

これは、 python path に実行時のモジュールパスまでしか含まれないことが原因です。
`sys.path.append(os.pardir)` とかを行う方法もありますが、
これは単純にファイル配置などによってしまうこともあり、 PEP8 に違反するようです。
そのため、実行時に `PYTHONPATH` を通す方法が良いようです。

```sh
export PYTHONPATH=path/to/root:$PYTHONPATH
# or 一時的に実行するだけであれば下記でもよい。
PYTHONPATH=path/to/root:$PYTHONPATH python package1/module1.py
```

実行時に最初からモジュールとして呼び出せば、実行位置が `PYTHONPATH` を通すだけであれば、
下記のように実行する方法もあります。
ただし、補完とかが効かなくなるので、呼び出すファイル名などを正しく書くのが難しくなります。

```sh
python -m package1.module1
```

- 参考資料
  - 2020.1.8 Qiita [【python再入門】親ディレクトリを経由したimportを行う方法][yokohama4580]

[yokohama4580]: https://qiita.com/yokohama4580/items/466a483ae022d264c8ee

