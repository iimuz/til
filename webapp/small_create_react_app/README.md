# Create React App を利用した小さなアプリケーション

## 環境構築

```sh
# バージョン確認
$ npm --version
$ node --version

# 環境構築
$ npx create-react-app ../small_create_react_app

# 開発関連のパッケージをdevDependenciesに移動
# dependenciesに"react", "react-dom"のみ残す。
$ npm ls --prod
small_create_react_app@0.1.0 /path/to/small_create_react_app
├─┬ react@17.0.2
│ ├─┬ loose-envify@1.4.0
│ │ └── js-tokens@4.0.0
│ └── object-assign@4.1.1
└─┬ react-dom@17.0.2
  ├── loose-envify@1.4.0 deduped
  ├── object-assign@4.1.1 deduped
  └─┬ scheduler@0.20.2
    ├── loose-envify@1.4.0 deduped
    └── object-assign@4.1.1 deduped
```

## Available Scripts

In the project directory, you can run:

- `npm start`
  - Runs the app in the development mode.
    Open [http://localhost:3000](http://localhost:3000) to view it in the browser.
  - The page will reload if you make edits.
    You will also see any lint errors in the console.
- `npm test`
  - Launches the test runner in the interactive watch mode.
  - See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.
- `npm run build`
  - Builds the app for production to the `build` folder.
    It correctly bundles React in production mode and optimizes the build for the best performance.
  - The build is minified and the filenames include the hashes.
    Your app is ready to be deployed!
  - See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.
- `npm run eject`
  - **Note: this is a one-way operation. Once you `eject`, you can’t go back!**
  - If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.
  - Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.
  - You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

- Code Splitting
  - This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)
- Analyzing the Bundle Size
  - This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)
- Making a Progressive Web App
  - This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)
- Advanced Configuration
  - This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)
- Deployment
  - This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)
- `npm run build` fails to minify
  - This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)

## 参考資料

- [新しい React アプリを作る][cra_new]

[cra_new]: https://ja.reactjs.org/docs/create-a-new-react-app.html
