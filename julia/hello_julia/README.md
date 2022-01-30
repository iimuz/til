# hello julia

Julia 環境を構築し、簡単な動作確認を行った時のメモ。

## 仮想環境の作成

Pkg モードで下記のコマンドで仮想環境を作成することができる。

- `generate package_name`: package_name ディレクトリの作成と管理用の Project.toml などが作成される。
- `activate package_name`: アクティベートしたいディレクトリをさして行う。(cd で移動済みであれば、`activate .`で良い。)
- 仮想環境から出るというコマンドは存在しないらしい。
- `instantiate`: 作成済みのリポジトリをクローンしてきたときに環境構築に利用する。

データ自体は何も設定しないと `~/.julia` 以下に作成されるらしい。
そのため、docker 環境で実験用などで使いまわす場合は、`~/.julia`をどこかホストからマウントした方がよさそう。コンテナを作り直すたびに毎回パッケージをダウンロードするため非常に遅くなる。

## Jupyter notebook

- julia 環境の REPL を設定しておけば、VSCode から特定の仮想環境で jupyter notebook が起動するみたい。
  下記コマンドで仮想環境を確認したら REPL と同じであった。

  ```jl
  import Pkg
  Pkg.status()
  ```

- jupyter notebook -> .jl については、 [Weave.jl][weave.jl]を利用すると変換できる。
  - 変換のサンプルコードは、[convert_to_jl.jl](src/convert_to_jl.jl)にある。
- .jl -> jupyter notebook についても Wave.jl で変換可能。
  VSCode の python で利用可能な interactive モードと同様に、`# %%`の部分でセルを分割できる。
  ただし、マークダウンを書かない場合、 `#+` を書いておかないと、すべてマークダウンセルが直前に付与される。
  したがって、単純にコードセルを作成したいときは、コメントとして下記の 2 行を付与すればよい。

  ```jl
  # %%
  #+
  hogehoge
  ```

- weave.jl 自体は、大雑把に分類すると markdown, tex, pandoc, html, restructuredtext, pdf, asciidoc あたりに変換できる。

[weave.jl]: https://github.com/JunoLab/Weave.jl
