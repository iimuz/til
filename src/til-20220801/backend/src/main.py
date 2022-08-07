import json
from io import StringIO
from typing import Tuple

import altair as alt
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from numpy import ndarray
from pandas import DataFrame

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """API動作確認用に単なるメッセージを返す."""
    return {"message": "Hello World"}


@app.get("/simple-line")
def simple_line():
    """sin波を作成してvega-liteのjson形式で返す."""
    x, y = _line_data(100)
    df = DataFrame({"x": x, "y": y})
    chart = alt.Chart(df).mark_line().encode(x="x", y="y")

    buff = StringIO()
    chart.save(buff, format="json")

    return json.loads(buff.getvalue())


@app.get("/simple-line-data")
def simple_line_data():
    """sin波を作成してdataのみ返す."""
    x, y = _line_data(100)

    return {"x": x.tolist(), "y": y.tolist()}


def _line_data(num: int) -> Tuple[ndarray, ndarray]:
    """sin波を生成する."""
    x = np.arange(num)
    y = np.sin(x / 5)

    return (x, y)
