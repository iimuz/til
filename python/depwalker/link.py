import bokeh.io as bkio
import bokeh.layouts as bklayouts
import bokeh.models as bkmodels
import bokeh.models.graphs as bkgraphs
import bokeh.palettes as bkpalettes
import bokeh.plotting as bkplotting
import bokeh.models.widgets as bkwidgets
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import re

from logging import getLogger
from typing import Dict, Set, Tuple

logger = getLogger(__name__)


def dep_tree_line_to_dict(line: str) -> Tuple[int, Dict]:
    """dependency walker のツリー部分のテキスト 1 行を辞書形式にする。

    Args:
        line (str): 1 行のテキスト

    Returns:
        Tuple[int, Dict]: インデントからわかる階層番号と情報

    Note:
        インデントがない状態の階層数を 0 とする。
    """
    legend = {
        " ": "None",
        "F": "Forwarded Module",
        "D": "Delay Load Module",
        "*": "Dynamic Module",
        "?": "Missing Module",
        "!": "Invalid Module",
        "E": "Import/Export Mismatch or Load Failure",
        "^": "Duplicate Module",
        "6": "64-bit Module",
    }
    space_num = 5
    pattern = re.compile(r"^(\s*)\[(.)(.)(.)\] (.*)$")
    result = re.match(pattern, line)
    if result is None:
        return (0, {})
    data = result.groups()
    if len(data) < 5:
        return (0, {})

    return (
        len(data[0]) // space_num,
        {
            "legend1": legend[data[1]],
            "legend2": legend[data[2]],
            "legend3": legend[data[3]],
            "name": data[4].strip(),
        },
    )


def main() -> None:
    """dependencty walker のツリー情報を解析する。
    """
    FILEPATH = "hoge.txt"
    df = to_df(FILEPATH)
    df = df[df["legend2"] != "Missing Module"]

    np.random.seed(0)

    graph = nx.from_pandas_edgelist(
        df, "name", "parent", edge_attr=None, create_using=nx.DiGraph()
    )
    nx.set_node_attributes(
        graph, df.drop_duplicates(subset="name").set_index("name").to_dict("index")
    )

    # グラフデータを bokeh 用に変換
    render = bkgraphs.from_networkx(graph, nx.spring_layout, scale=1, center=(0, 0))
    render.node_renderer.glyph = bkmodels.Circle(
        size=8, fill_color=bkpalettes.Spectral4[0]
    )

    # グラフ初期設定
    p = bkplotting.figure(
        tools=(
            "pan,box_zoom,lasso_select,box_select,poly_select"
            ",tap,wheel_zoom,reset,save,zoom_in"
        ),
        title="dependency link",
        x_range=(-1.1, 1.1),
        y_range=(-1.1, 1.1),
    )
    tooltips = [
        ("name", "@index"),
        ("legend1", "@legend1"),
        ("legend2", "@legend2"),
        ("legend3", "@legend3"),
    ]
    p.add_tools(bkmodels.HoverTool(tooltips=tooltips))
    p.renderers.append(render)

    # データ表示データテーブル
    data_table = bkwidgets.DataTable(
        source=render.node_renderer.data_source,
        columns=[
            bkwidgets.TableColumn(field=column, title=column)
            for column in ["index", "legend1", "legend2", "legend3"]
        ],
        fit_columns=True,
    )

    dependency_source = bkmodels.ColumnDataSource(df)
    dependency_table = bkwidgets.DataTable(
        source=dependency_source,
        columns=[
            bkwidgets.TableColumn(field=column, title=column)
            for column in ["name", "parent", "legend1", "legend2", "legend3"]
        ],
        fit_columns=True,
    )

    # 依存関係の始点と終点を設定するテキストボックス
    target_text = bkwidgets.TextInput(value="None", title="Target Module")
    source_text = bkwidgets.TextInput(value="Input Module Name", title="Source Module")
    cutoff_text = bkwidgets.TextInput(value="3", title="Number of Cutoff")

    # 実行ボタン
    def execute_callback(event) -> None:
        nonlocal render
        nonlocal dependency_source
        np.random.seed(0)

        all_pathes = nx.all_simple_paths(
            graph,
            source=f"{source_text.value}",
            target=f"{target_text.value}",
            cutoff=int(f"{cutoff_text.value}"),
        )
        pathes: Set = set()
        for path in all_pathes:
            pathes |= set(path)
        subgraph = graph.subgraph(pathes)
        render = bkgraphs.from_networkx(
            subgraph, nx.spring_layout, scale=1, center=(0, 0)
        )
        render.node_renderer.glyph = bkmodels.Circle(
            size=8, fill_color=bkpalettes.Spectral4[0]
        )
        p.renderers.clear()
        p.renderers.append(render)
        data_table.source = render.node_renderer.data_source

        x, y = zip(*render.layout_provider.graph_layout.values())
        render.node_renderer.data_source.data["x"] = x
        render.node_renderer.data_source.data["y"] = y
        labels = bkmodels.LabelSet(
            x="x", y="y", text="index", source=render.node_renderer.data_source
        )
        p.renderers.append(labels)

        dependency_df = df[
            df["name"].isin(pathes) & df["parent"].isin(pathes)
        ].drop_duplicates(subset=["name", "parent"])
        dependency_source = bkmodels.ColumnDataSource(dependency_df)
        dependency_table.source = dependency_source

    execute_button = bkwidgets.Button(label="execute", button_type="success")
    execute_button.on_click(execute_callback)

    # 検索結果をクリアするボタン
    def clear_callback(event) -> None:
        np.random.seed(0)
        render = bkgraphs.from_networkx(graph, nx.spring_layout, scale=1, center=(0, 0))
        render.node_renderer.glyph = bkmodels.Circle(
            size=8, fill_color=bkpalettes.Spectral4[0]
        )
        p.renderers.clear()
        p.renderers.append(render)
        # p.renderers.pop(0)
        data_table.source = render.node_renderer.data_source

    clear_button = bkwidgets.Button(label="clear", button_type="success")
    clear_button.on_click(clear_callback)

    # レイアウト
    execute_button_area = bklayouts.layout(
        [[execute_button, clear_button]], sizing_mode="stretch_width"
    )
    execute_area = bklayouts.layout(
        [target_text, source_text, cutoff_text, execute_button_area, dependency_table],
        sizing_mode="stretch_width",
    )
    operation_area = bklayouts.layout(
        [data_table, execute_area], sizing_mode="stretch_both"
    )
    layout = bklayouts.layout([[p, operation_area]], sizing_mode="stretch_both")

    bkio.curdoc().add_root(layout)


def to_df(filepath: str) -> pd.DataFrame:
    """dependency walker の出力するファイルからツリー部分のみをデータフレームへ変換する。

    Args:
        filepath (str): ファイルパス

    Returns:
        pd.DataFrame: データフレーム
    """
    START_SPLITER = "*| Module Dependency Tree |*"
    END_SPLITER = "*| Module List |*"
    parents = {-1: "None"}

    def line_to_seq(line: str) -> Dict:
        nonlocal parents

        layer_index, data = dep_tree_line_to_dict(line.rstrip())
        if "name" not in data:
            return {}

        parents[layer_index] = data["name"]
        data["parent"] = parents[layer_index - 1]
        return data

    df = pd.DataFrame()
    with open(filepath) as f:
        for line in f:
            if START_SPLITER in line:
                break
        for line in f:
            if line == "\n":
                break
        df = pd.DataFrame(
            [
                line_to_seq(line)
                for line in itertools.takewhile(
                    lambda line: END_SPLITER not in line, (line for line in f)
                )
            ]
        )
    df = df.dropna()

    return df


main()
