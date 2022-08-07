import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';

import { Hello } from './hello';
import { SimpleLineCompile } from './simple-line-compile';
import { SimpleLineJson } from './simple-line-json';

const App = (): JSX.Element => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path={'/'} element={<Home />} />
        <Route path={'/message'} element={<Hello />} />
        <Route path={'/simple-line-json'} element={<SimpleLineJson />} />
        <Route path={'/simple-line-compile'} element={<SimpleLineCompile />} />
      </Routes>
    </BrowserRouter>
  );
};

const container = document.getElementById('root');
if (!container) throw new Error('Failed to find the root element');
const root = createRoot(container);
root.render(<App />);

// 各実装へのリンク一覧を記載
const Home = (): JSX.Element => {
  return (
    <div>
      <ul>
        <li>
          <Link to='/message'>Message</Link>: バックエンドに接続して文字列取得できるか確認。
        </li>
        <li>
          <Link to='/simple-line-json'>Simple line json</Link>: バックエンドのAltairでjson出力してvega-embedで描画。
        </li>
        <li>
          <Link to='/simple-line-compile'>Simple line compile</Link>:
          バックエンドからデータだけ取得してvega-liteでcompileしてVegaで描画。
        </li>
      </ul>
    </div>
  );
};
