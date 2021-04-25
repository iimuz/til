# git

git 関連のスクリプトサンプルです。

## git の最新バージョンインストール

ubuntu 環境で最新の git をインストールするためには、ppa を追加する必要があります。

```sh
sudo add-apt-repository ppa:git-core/ppa
sudo apt update
sudo apt upgrade
```

(参考資料)

- 2020/6/6 [Ubuntu で Git の最新を使う][cointoss1973]

[cointoss1973]: https://qiita.com/cointoss1973/items/1c01837e65b937fc0761

## sparse-checkout

`git sparse-checkout`を利用することで、Git リポジトリの一部だけをクローンすることができます。
既存のリポジトリが下記のような構造を持つときに directoryA のみを取得します。
あくまでも repository/directoryA の構造をクローンするため、 directoryA をリポジトリのトップディレクトリのようにはクローン出来ないようです。

```txt
|- repository
  |- directoryA
  |- direcotryB
```

下記のようにして sparse-checkout できます。

```sh
git clone --filter=blob:none --no-checkout https://example.com/hoge/geho.git
cd geho
git sparse-checkout init --cone
git sparse-checkout add directoryA
```

(参考資料)

- 2020/6/4 [モノリポ時代に知っておくと便利な「git sparse-checkout」][kakakakakku]

[kakakakakku]: https://kakakakakku.hatenablog.com/entry/2020/06/04/104940

### worktree との合わせ技

worktree を利用することで別のフォルダに別ブランチの作業ディレクトリを作成することができます。
sparse-checkout と合わせることができるため、下記のようにすると worktree 内だけ sparse-checkout を有効にすることができます。

```sh
git worktree add --no-checkout -b feat/sparse-checkout /path/to/working/directory branch/name
cd /path/to/working/directory
git sparse-checkout init --cone
git sparse-checkout add directoryA
```

(参考資料)

- 2018/1/24 [Creating sparse checkout in a new linked git worktre][54fd6a226955dc427bb25d5be37b4b0a]

[54fd6a226955dc427bb25d5be37b4b0a]: https://public-inbox.org/git/54fd6a226955dc427bb25d5be37b4b0a.squirrel@mail.jessiehernandez.com/t/
