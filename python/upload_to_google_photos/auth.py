import logging

import click

import google_oauth2
import google_photos


@click.group()
@click.option(
    '--client',
    type=str,
    default='./_test/client_id.json',
    help='Input path for client id json.'
)
@click.option(
    '--credential',
    type=str,
    default='./_test/credentials.json',
    help='Input output path for credential json.'
)
@click.pass_context
def cli(ctx, client: str, credential: str):
    """ Google Photos Libraryを利用するコマンド
    """
    ctx.obj['client'] = client
    ctx.obj['credential'] = credential


def create_logger() -> logging.Logger:
    """ ログ出力のための設定
    """
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


@click.command()
@click.pass_context
def login(ctx) -> None:
    """ Google認証のみを行う
    """
    logger = create_logger()
    google_oauth2.get_authorized_service(
        ctx.obj['client'], ctx.obj['credential'], logger)


if __name__ == "__main__":
    cli.add_command(login)
    cli(obj={})
