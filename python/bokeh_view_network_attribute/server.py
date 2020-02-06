import bokeh.io as bkio
import bokeh.layouts as bklayouts
import bokeh.models as bkmodels
import bokeh.models.graphs as bkgraphs
import bokeh.palettes as bkpalettes
import bokeh.plotting as bkplotting
import networkx as nx
import numpy as np

from logging import getLogger

logger = getLogger(__name__)


def main() -> None:
    """メイン処理

    Note:
        from_networkx ではランダム値によりグラフ中のノード位置が決定する。
        そのため、 numpy の random seed を固定することで、毎回同じ形のグラフが出力されるようにする。
    """
    # グラフデータ
    graph = nx.karate_club_graph()

    # エッジの色を設定
    SAME_CLUB_COLOR, DIFFERENT_CLUB_COLOR = "black", "red"
    edge_attrs = {}
    for start_node, end_node, _ in graph.edges(data=True):
        edge_attrs[(start_node, end_node)] = (
            SAME_CLUB_COLOR
            if graph.nodes[start_node]["club"] == graph.nodes[end_node]["club"]
            else DIFFERENT_CLUB_COLOR
        )
    nx.set_edge_attributes(graph, edge_attrs, "edge_color")

    # グラフ初期設定
    p = bkplotting.figure(
        tools=(
            "pan,box_zoom,lasso_select,box_select,poly_select"
            ",tap,wheel_zoom,reset,save,zoom_in"
        ),
        title="Networkx Integration Demonstration",
        x_range=(-2.1, 2.1),
        y_range=(-2.1, 2.1),
    )
    p.add_tools(bkmodels.HoverTool(tooltips=[("index", "@index"), ("club", "@club")]))

    np.random.seed(0)
    render = bkgraphs.from_networkx(graph, nx.spring_layout, scale=2, center=(0, 0))
    render.node_renderer.glyph = bkmodels.Circle(
        size=15, fill_color=bkpalettes.Spectral4[0]
    )
    render.edge_renderer.glyph = bkmodels.MultiLine(
        line_color="edge_color", line_alpha=0.6, line_width=2
    )
    p.renderers.append(render)

    # レイアウト
    layout = bklayouts.layout([p], sizing_mode="stretch_both")

    bkio.curdoc().add_root(layout)


main()
