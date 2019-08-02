import bokeh.io as bkhio
import bokeh.layouts as bkhlayouts
import bokeh.models.graphs as bkhgraphs
import bokeh.plotting as bkhplotting
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
    np.random.seed(0)

    graph = nx.karate_club_graph()
    graph = bkhgraphs.from_networkx(graph, nx.spring_layout, scale=2, center=(0, 0))

    # グラフ初期設定
    p = bkhplotting.figure(
        tools=(
            "pan,box_zoom,lasso_select,box_select,poly_select"
            ",tap,wheel_zoom,reset,save,zoom_in"
        ),
        title="Networkx Integration Demonstration",
        x_range=(-2.1, 2.1),
        y_range=(-2.1, 2.1),
    )
    p.renderers.append(graph)

    # レイアウト
    layout = bkhlayouts.layout([p], sizing_mode="stretch_both")

    bkhio.curdoc().add_root(layout)


main()
