"""htmlを出力するための実行スクリプト."""
# default packages
import dataclasses as dc
import logging

# third party packages
import jinja2

# logger
_logger = logging.getLogger(__name__)


@dc.dataclass
class Config:
    pass


def main(config: Config):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(".", encoding="utf8"))

    tmpl = env.get_template("index.j2")
    html = tmpl.render(shop="テスト")

    _logger.info(html)


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)

        _config = Config()
        main(_config)
    except Exception as e:
        _logger.exception(e)
