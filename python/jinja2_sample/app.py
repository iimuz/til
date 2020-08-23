"""htmlを出力するための実行スクリプト."""
# default packages
import dataclasses as dc
import logging
import pathlib

# third party packages
import jinja2

# logger
_logger = logging.getLogger(__name__)


@dc.dataclass
class Config:
    outdir: str = "path/to/output"


def main(config: Config):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(".", encoding="utf8"))

    tmpl = env.get_template("index.j2")
    html = tmpl.render(comment="hoge")

    outdir = pathlib.Path(config.outdir)
    outdir.mkdir(exist_ok=True)
    filepath = outdir.joinpath("index.html")
    with open(str(filepath), "w") as f:
        f.write(html)


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)

        _config = Config(outdir="output")
        main(_config)
    except Exception as e:
        _logger.exception(e)
