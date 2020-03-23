# マハラノビス距離を計算するサンプル

マハラノビス距離をサンプルコードです。

- `mahalanobis_distance.py`: numpy のみで[マハラノビス距離][wiki_en]を計算するサンプル
- `mahalanobis_scipy.py`: [scipy の関数][scipy]を利用したサンプル
- `md2.py`: 品質工学の MD 値[^md2]を計算するサンプル

[^md2]: 田口 玄一, ラジェッシュ J, (訳)手島 昌一, [多変量診断の新たな流れ][md2], 品質工学, 2001, 9 巻, 4 号, p. 74-95, 公開日 2016/08/31, Online ISSN 2189-9320, Print ISSN 2189-633X.

[md2]: https://www.jstage.jst.go.jp/article/qes/9/4/9_74/_pdf/-char/ja
[scipy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html
[wiki_en]: https://en.wikipedia.org/wiki/Mahalanobis_distance
