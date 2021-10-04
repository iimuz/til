# fronend フレームワークを利用しない小型プロジェクトの構成サンプル

vue.js や ReactJS を利用せずに簡易に実現するプロジェクトのサンプル。
下記の機能を有する用に作成している。

- frontend 側で typescript/html を作成。
- backend 側で frontend 側で作成した js/html を配信。
- fronend 側の js で XMLHttpRequest を利用し get を行い, backend 側で get を受け付ける。
- get の query を利用して任意の内容をページに表示する。
- html は分割して共通部分を複数個所に書かなくてよいようにする。

## 実行

開発時は下記のように実行しておけば、forntend, backend のいずれを書き換えても変更が適用される。

```sh
# 下記2つをあらかじめ実行しておく
npm run watch
FLASK_APP=main FLASK_ENV=development poetry run flask run
```

## frontend

- ts コンパイル: `npm run build`
- ts コンパイル(変更検知版): `npm run watch`
- html の生成結果確認用サーバ立ち上げ: `npm run start`
- html テンプレートの対応: `npm i --save-dev html-loader html-webpack-plugin`
  - html-loader を利用して`src/html`のファイルを dist/html 下に作成する。
- html を簡易に確認: `npm i --save-dev webpack-dev-server`
  - 特定フォルダ下の html を簡易サーバで閲覧できるようにする。
- webpack.config.js の output に publicPath を記載することで、html に記述される js の場所などを特定フォルダをトップとして記述することができる。
- webpack.config.js で chunks を指定することで、各ページに必要なスクリプトを追加するように修正
- javascript で query string を取得するには、`window.location.search.slice(1)`とすればよい。

## backend

- サーバの実行: `FLASK_APP=main FLASK_ENV=development poetry run flask run`
- `FLASK_ENV=development`指定することでコード変更による自動修正を追加。
- flask app.route に methods を追加して GET とすることで GET 指定。
- flask で query string を取得するには、 `flask.request.query_string`を利用し、decode すればよい。

## 参考資料

- 2019-04-19 [TypeScript & ESLint/Prettier & webpack & Jest で最小構成のプロジェクトを作る][yamash]
  - eslint/prettier の環境構築方法が掲載されている。
- 2020-08-07 [webpack で共通 HTML をテンプレート化][ajike]
  - html-loader を利用した webpack での html 出力を記載している。
- 2021-05-12 [最新版 TypeScript+webpack 5 の 環境構築まとめ][ics16329]
  - webpack と typescript の環境を構築する最小限の構成が載っている。
- 2021-09-27 [最新版で学ぶ webpack 5 入門 JavaScript のモジュールバンドラ][ics12140]
  - webpack を利用してソースマップを有効にする方法、dev-server の立て方が説明されている。

[ajike]: https://ajike.github.io/webpack_html/
[ics16329]: https://ics.media/entry/16329/
[ics12140]: https://ics.media/entry/12140/
[yamash]: https://yamash.hateblo.jp/entry/2019/04/19/200000
