import React, { useEffect } from 'react';
import { parse, Spec, View } from 'vega';
import { VisualizationSpec } from 'vega-embed';
import { compile } from 'vega-lite';

import { HOST } from './fetcher';

type LineData = { x: number[]; y: number[] };

// バックエンドからデータのみ取得してVega-liteからVegaで描画
export const SimpleLineCompile = (): JSX.Element => {
  const container = 'view';
  useEffect(() => {
    getSimpleLineData()
      .then((data) => render(data, `#${container}`))
      .catch((err) => console.error(err));
  });

  return <div id={container}></div>;
};

// バックエンドから描画するデータのみ取得
async function getSimpleLineData(): Promise<LineData> {
  const res = await fetch(HOST + '/simple-line-data');
  if (!res.ok) {
    const defaultData: LineData = { x: [], y: [] };
    return defaultData;
  }
  const data = await res.json();

  return data;
}

// dataをvega-liteの記法を利用し、vega specに変換している。
async function render(data: LineData, container: string): Promise<View> {
  const values = data.x.map((x, index) => ({ x: x, y: data.y[index] }));

  // vega-lite specからvega specに変換
  const spec: VisualizationSpec = {
    config: { view: { continuousWidth: 400, continuousHeight: 300 } },
    mark: 'line',
    encoding: { x: { field: 'x', type: 'quantitative' }, y: { field: 'y', type: 'quantitative' } },
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    data: { values: values },
  };
  const vegaSpec = compile(spec).spec;

  // vega specから描画
  const view = new View(parse(vegaSpec), { renderer: 'canvas', container: container, hover: true });

  return view.runAsync();
}
