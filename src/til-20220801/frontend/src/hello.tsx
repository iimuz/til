import React, { useEffect, useState } from 'react';
import { HOST } from './fetcher';

// バックエンドに問い合わせて単純な文字列を表示。
export const Hello = (): JSX.Element => {
  const [message, setMessage] = useState('');
  useEffect(() => {
    getHello()
      .then((msg) => setMessage(msg))
      .catch((err) => console.log(err));
  }, [message]);

  return <div>{message}</div>;
};

// バックエンドに接続して文字列を取得。
async function getHello(): Promise<string> {
  const res = await fetch(HOST);
  if (!res.ok) return 'error';
  const data = await res.json();
  const message = data['message'];

  return message;
}
