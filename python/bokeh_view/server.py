import bokeh
import logging
import numpy as np
import pandas as pd
import umap

from bokeh.events import ButtonClick
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models.widgets import (
    Button,
    Select,
    RadioGroup,
    TextInput,
    TableColumn,
    DataTable,
)
from bokeh.layouts import Column, Row
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from logging import getLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from typing import Dict, List

logger = getLogger(__name__)


def analysis_method() -> Dict:
    """解析手法名と解析実行のペアを返す

    Returns:
        Dict: 解析手法名 -> 解析用関数
    """
    method = {
        "PCA": lambda data: PCA(n_components=2).fit_transform(data),
        "tSNE": lambda data: TSNE(n_components=2).fit_transform(data),
        "MDS": lambda data: MDS(n_jobs=4).fit_transform(data),
        "UMAP": lambda data: umap.UMAP().fit_transform(data),
    }
    return method


def create_hover(names: List[str]) -> HoverTool:
    """hover表示をする設定を返す

    Args:
        names (List[str]): hover表示を行う列名

    Returns:
        HoverTool: hover表示を行うツールの実体
    """
    tooltips = [("index", "$index")]
    tooltips.extend([(name, f"@{name}") for name in names])
    hover = HoverTool(tooltips=tooltips)

    return hover


def create_mapper(df: pd.DataFrame, field_name: str) -> Dict:
    """指定した DataFrame の field_name を利用したカラー用マッパーを返す

    Args:
        df (pd.DataFrame): 対象とするデータフレーム
        field_name (str): カラーを調整するデータ列名

    Returns:
        Dict: 調整したマッパー
    """
    mapper = linear_cmap(
        field_name=field_name,
        palette=bokeh.palettes.Viridis256,
        low=min(df[field_name].values),
        high=max(df[field_name].values),
    )

    return mapper


def to_source_data_from(df: pd.DataFrame, result: np.ndarray) -> Dict:
    """データフレームと次元削減結果からソースデータを作成する

    Args:
        df (pd.DataFrame): 全データ
        result (np.ndarray): 次元削減結果(2次元以上)

    Returns:
        Dict: ソースデータ
    """
    data = {"ID": df.index.values, "0": result[:, 0], "1": result[:, 1]}
    data.update({column: df[column] for column in df.columns})

    return data


def to_datatable_columns_from(df: pd.DataFrame) -> List:
    """データテーブル用の列名リストを生成する

    Args:
        df (pd.DataFrame): データフレーム

    Returns:
        List: 列名リスト
    """
    columns = [
        TableColumn(field=column, title=column, width=100) for column in df.columns
    ]
    columns.append(TableColumn(field="ID", title="ID", width=100))

    return columns


def main() -> None:
    """メイン処理

    Returns:
        None: None
    """
    # データソースの初期設定
    source = ColumnDataSource(data=dict(length=[], width=[]))
    source.data = {"0": [], "1": []}
    df = pd.DataFrame()

    # CSVファイル設定テキストボックス
    csv_input = TextInput(value="default.csv", title="Input CSV")

    # 可視化手法選択ラジオボタン
    method_group = analysis_method()
    method_radio_group = RadioGroup(labels=list(method_group.keys()), active=0)

    # グラフ初期設定
    p = figure(
        tools=(
            "pan,box_zoom,lasso_select,box_select,poly_select"
            ",tap,wheel_zoom,reset,save,zoom_in"
        ),
        title="Analyze Result",
        plot_width=1000,
        plot_height=800,
    )
    p.circle(x="0", y="1", source=source)

    # データ表示データテーブル
    data_table = DataTable(
        source=source, columns=[], width=600, height=500, fit_columns=False
    )

    # 色設定項目用選択セレクトボックス
    def color_select_callback(attr, old, new) -> None:
        mapper = create_mapper(df, new)
        p.circle(x="0", y="1", source=source, line_color=mapper, color=mapper)

    color_select = Select(title="color:", value="0", options=[])
    color_select.on_change("value", color_select_callback)

    # 解析実行ボタン
    def execute_button_callback_inner(evnet):
        nonlocal df
        df = pd.read_csv(csv_input.value, index_col=0)
        result = method_group[method_radio_group.labels[method_radio_group.active]](df)
        source.data = to_source_data_from(df, result)
        data_table.columns = to_datatable_columns_from(df)
        mapper = create_mapper(df, df.columns.values[0])
        p.circle(x="0", y="1", source=source, line_color=mapper, color=mapper)
        p.add_tools(create_hover(["ID", df.columns.values[0]]))
        color_select.options = list(df.columns)

    execute_button = Button(label="Execute", button_type="success")
    execute_button.on_event(ButtonClick, execute_button_callback_inner)

    # レイアウト
    operation_area = Column(
        csv_input, method_radio_group, execute_button, color_select, data_table
    )
    layout = Row(p, operation_area)
    curdoc().add_root(layout)


main()
