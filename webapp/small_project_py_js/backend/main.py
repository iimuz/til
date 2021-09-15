import flask
from flask import Flask

app = Flask(
    __name__,
    static_folder="../frontend/dist/static",
    template_folder="../frontend/dist",
)


@app.route("/", defaults={"path": ""})
def index(path: str):
    return flask.render_template("index.html")


if __name__ == "__main__":
    app.run()
