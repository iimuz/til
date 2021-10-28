# React と FastAPI を利用した小型プロジェクトの構成サンプル

React と FastAPI を利用して小型の SPA アプリケーションを作成するサンプル。

## 実行方法

## 環境構築方法

## frontend

- `npm start`: runs the app in the development mode.
- `npm test`: launches the test runner in the interactive watch mode.
- `npm run build`: builds the app for production to the `build` forlder.

## backend

- `poetry run uvicorn main::app --reload`: start server
  - `localhost:8000/docs`: swagger ui を利用した api ドキュメントの確認
  - `localhost:8000/redoc`: redoc を利用した api ドキュメントの確認
- [静的ファイルを配信][static-files]するため aiofiles を追加
  - [FileResponse を利用してルートで html を返す][fastapi130]

[fastapi130]: https://github.com/tiangolo/fastapi/issues/130
[static-files]: https://fastapi.tiangolo.com/ja/tutorial/static-files/

## 参考資料
