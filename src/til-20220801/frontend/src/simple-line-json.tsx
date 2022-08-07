import React, { useEffect } from 'react';

import embed, { VisualizationSpec } from 'vega-embed';
import { HOST } from './fetcher';

// バックエンドからVega-embedで描画するためのjsonが送られてきた場合の描画
export const SimpleLineJson = (): JSX.Element => {
  const canvasName = 'vis';

  useEffect(() => {
    getSimpleLineJson()
      .then((data) => embed(`#${canvasName}`, data))
      .catch((err) => console.log(err));
  }, []);

  return <div id={canvasName}></div>;
};

// バックエンドからvega-liteで描画するためのjsonを取得
async function getSimpleLineJson(): Promise<VisualizationSpec> {
  const res = await fetch(HOST + '/simple-line');
  if (!res.ok) return {};
  const spec: VisualizationSpec = await res.json();

  return spec;
}
