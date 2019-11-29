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
def cli(ctx, client: str, credential: str) -> None:
    """ Google Photos Libraryのアルバム操作を行うコマンド群
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
def list(ctx) -> None:
    """ アルバムリストを取得する
    """
    logger = create_logger()

    service = google_oauth2.get_authorized_service(
        ctx.obj['client'],
        ctx.obj['credential'],
        logger)
    client = google_photos.GooglePhots(service)

    album_list = client.get_album_list()
    for album in album_list['albums']:
        logger.info(album)


@click.command()
@click.argument(
    'name',
    type=str,
)
@click.pass_context
def new(ctx, name: str) -> None:
    """ 新規アルバムを作成する
    """
    logger = create_logger()

    service = google_oauth2.get_authorized_service(
        ctx.obj['client'],
        ctx.obj['credential'],
        logger)
    client = google_photos.GooglePhots(service)

    album_id = client.create_new_album(name)
    logger.info(f"album id: {album_id}")


@click.command()
@click.argument(
    'name',
    type=str,
)
@click.option(
    '--id',
    is_flag=True,
    help='search using id.',
)
@click.pass_context
def search(ctx, name: str, id: int) -> None:
    """ 既存のアルバムを検索する
    """
    logger = create_logger()

    service = google_oauth2.get_authorized_service(
        ctx.obj['client'],
        ctx.obj['credential'],
        logger)
    client = google_photos.GooglePhots(service)

    target = 'id' if id else 'title'
    album_list = client.get_album_list()
    logger.info('-' * 5 + ' albums ' + '-' * 5)
    for album in album_list['albums']:
        if not album[target] == name:
            continue
        logger.info(f"album list: {album}")
    logger.info('-' * 10)


if __name__ == "__main__":
    cli.add_command(new)
    cli.add_command(list)
    cli.add_command(search)
    cli(obj={})
