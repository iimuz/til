# React.js + Heatmap

React.js で Heatmap らしき表示を実装する簡易サンプル。

```sh
npm i -D css-loader style-loader  # css利用
npm i -D css-modules-typescript-loader  # typescriptでcssを利用する場合
```

## 参考資料

- [React Heatmap Table][link00]: 参考にしたコード
- [Can't import CSS/SCSS modules. TypeScript says "Cannot Find Module"][link00]

  - css をインポートした際にエラーが発生したので対応方法を調査。
    下記の記述で対応できた。

    ```js
    module.exports = {
      resolve: {
        extensions: ['.ts', '.tsx', '.js', '.css'],
      },
      module: {
        rules: [
          { test: /\.tsx?$/, loader: 'ts-loader' },
          {
            test: /\.scss$/,
            use: [
              { loader: 'style-loader' },
              { loader: 'css-modules-typescript-loader' },
              { loader: 'css-loader', options: { modules: true } },
            ],
          },
        ],
      },
    };
    ```

[link00]: https://codepen.io/tkim90/pen/wvMabdr
[link01]: https://stackoverflow.com/questions/40382842/cant-import-css-scss-modules-typescript-says-cannot-find-module
