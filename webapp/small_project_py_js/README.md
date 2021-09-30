# fronend フレームワークを利用しない小型プロジェクトの構成サンプル

vue.js や ReactJS を利用せずに簡易に実現するプロジェクトのサンプル。

## frontend

- ts コンパイル: `npm run build`

## backend

- サーバの実行: `FLASK_APP=main poetry run flask run`

## 参考資料

- 2021-05-12 [最新版 TypeScript+webpack 5 の 環境構築まとめ][ics]
  - webpack と typescript の環境を構築する最小限の構成が載っている。
- 2019-04-19 [TypeScript & ESLint/Prettier & webpack & Jest で最小構成のプロジェクトを作る][yamash]
  - eslint/prettier の環境構築方法が掲載されている。

[ics]: https://ics.media/entry/16329/
[yamash]: https://yamash.hateblo.jp/entry/2019/04/19/200000
