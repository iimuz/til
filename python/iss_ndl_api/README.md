# 国会図書館サーチ API

国会図書館サーチの API を利用して検索するサンプルです。
サーチ結果は XML で返ってくるため、pandas.DataFrame へ変換して表示するところまでを実装しています。
国会図書館サーチ API は、複数のプロトコルが用意されていますが、今回は OpenSearch の方法を利用しています。

主な検索キーとしては下記があるようです。

- dpid: データプロバイダ ID
- title: 書籍タイトル
- creator: 著者
- from: 検索期間で開始出版年月日
- until: 検索期間で終了出版年月日
- mediatype: 指定しないと全てのタイプが検索対象となる。
  - 1: 本
  - 2: 記事・論文
  - 3: 新聞
  - 4: 児童書
  - 5: レファレンス情報
  - 6: デジタル資料
  - 7: その他
  - 8: 障碍者向け飼料
  - 9: 立法情報
- cnt: 出力レコード上限
- idx: レコード取得開始位置

返ってくる XML は item タグが書籍 1 本に相当し、その下にタイトルなどが含まれています。

- root
  - channel
    - item
      - title
      - author
    - item
      - title
      - author
    - ...

## Usage

下記コマンドで予め決めたパラメータで検索するサンプルを動かします。

```sh
pipenv install --skip-lock
pipenv run python search.py
```

## 参考資料

- [国会図書館サーチ][iss-ndl]
- [国会図書館サーチ ＡＰＩ で書籍情報をまとめて取得－python][ailaby]
- 2019.3.11: [国立国会図書館 API を使いやすくするための Python ライブラリを作成しました][shimakaze]
- 2016.10.29: [Python で国立国会図書館サーチ API を使用][noco]

[ailaby]: http://ailaby.com/ndl_search/
[iss-ndl]: https://iss.ndl.go.jp/
[shimakaze]: https://note.mu/shimakaze_soft/n/nfd4a6bf7e79d
[noco]: https://qiita.com/noco/items/c2d1f4308ee7a50c462e
