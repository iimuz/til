import logging

import click

import google_oauth2
import google_photos


@click.command()
@click.option(
    '--create',
    type=str,
    help='create new album.'
)
@click.option(
    '--id',
    type=int,
    help='get album name from album id.'
)
@click.option(
    '--name',
    type=str,
    help='get alubm id from album name.'
)
@click.pass_context
def album(ctx, create: str, id: int, name: str):
    """ アルバムを操作するためのコマンド
    """
    logger = create_logger()

    service = google_oauth2.get_authorized_service(
        ctx.obj['client'],
        ctx.obj['credential'],
        logger)
    client = google_photos.GooglePhots(service)

    if create is not None:
        client.create_new_album(name)
        logger.info(f"create new album: {name}")
        return

    if id is not None:
        logger.error('id option is not implemented.')
        return

    if name is not None:
        logger.error('name option is not implemented.')
        return

    album_list = client.get_album_list()
    logger.info(f"album list: {album_list}")


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


@click.command()
@click.argument('path')
@click.option(
    '--alubmId',
    type=int,
    help=''
)
def upload(path: str, alubmId: int) -> None:
    """ 画像をアップロードする
    """
    logger = create_logger()

    # service = google_oauth2.get_authorized_service()
    # client = google_photos.GooglePhots(service)

    # album_list = client.get_album_list()
    # logger.info(f"album list: {album_list}")
    logger.info(f"{path}")

    # album_name = "test"
    # client.create_new_album(album_name)
    # logger.info(f"create new album: {album_name}")

    # IMAGE_PATH = '_test/enable_photos_api.png'
    # response = client.upload_image(IMAGE_PATH)
    # logger.info(response)
    # logger.info(response['newMediaItemResults'][0]['status'])


if __name__ == "__main__":
    cli.add_command(album)
    cli.add_command(login)
    cli.add_command(upload)
    cli(obj={})
