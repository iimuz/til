import bokeh.layouts as bkhlayouts
import bokeh.io as bkhio
import bokeh.models as bkhmodels
import bokeh.models.widgets as bkhwidgets
import pandas as pd


def main() -> None:
    df = pd.read_csv("../_data/iris.csv")

    # データソースの初期設定
    source = bkhmodels.ColumnDataSource(df)

    # データ表示データテーブル
    data_table = bkhwidgets.DataTable(
        source=source,
        columns=[
            bkhwidgets.TableColumn(field=column, title=column) for column in df.columns
        ],
        fit_columns=True,
    )

    # レイアウト
    layout = bkhlayouts.column(data_table, sizing_mode="stretch_both")
    bkhio.curdoc().add_root(layout)


main()
