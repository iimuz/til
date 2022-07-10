# Minimal webpack + typescript + react

Webpack + React + Typescript の環境をなるべく依存関係を少なくして実現するサンプルコード。

環境構築にあたってインストールに利用したコマンドは下記の通り。
複数行のコマンドをまとめても問題はない。

```sh
# webpack, typescript, react を追加
npm init
npm i -D webpack webpack-cli typescript ts-loader html-webpack-plugin
npm i -S react react-dom
npm i -D @types/react @types/react-dom  # 型情報の追加

# 開発環境用の設定を追加
## 簡易サーバー
npm i -D webpack-dev-server
npm i -D
## eslint, prettierの設定追加
npm i -D \
  @types/node \
  @typescript-eslint/eslint-plugin \
  @typescript-eslint/parser \
  eslint \
  eslint-config-prettier \
  eslint-config-react-app \
  eslint-plugin-import \
  prettier
```

- htmlをsrcディレクトリからbuildディレクトリへコピーするために html-webpack-pluginを利用
- 開発時にサーバーを立ち上げるために webpack-dev-server も利用
- 開発用に eslint, prettier の設定を追加
  - `.eslintrc.json`: eslint 用の設定
  - `.prettierrc.json`: prettier 用の設定

下記のコマンドが利用できるように設定している。

- `npm run build`: buildディレクトリ以下にファイル群を作成
- `npm run format`: ファイルの整形チェックを行い修正
- `npm run lint`: ファイルの整形チェック
- `npm run production`: 最終成果物用のビルド
- `npm run start`: サーバーを起動し、ファイル変更を検知してリロード

## 参考資料

- [最新版TypeScript+webpack 5の 環境構築まとめ][link00]: 主に参考にさせてもらったサイト。

[link00]: https://ics.media/entry/16329/#webpack-ts-react
