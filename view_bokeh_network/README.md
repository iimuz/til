# Visualize network graph using bokeh

Bokeh を利用してネットワークグラフを表示するサンプルです。

- [Visualizing Network Graphs: Networkx Integration][bokeh_graph]

[bokeh_graph]: https://bokeh.pydata.org/en/latest/docs/user_guide/graph.html

## Usage

```sh
bokeh serve server.py
```

ログ情報を出力する場合は、 bokeh の実行時にログレベルを設定する必要があります。

```sh
bokeh serve server.py --log-level info
```
