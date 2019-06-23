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
@click.argument('path')
@click.option(
    '--album',
    type=str,
    help=''
)
@click.option(
    '--id',
    is_flag=True,
    help='set album using id.',
)
@click.pass_context
def upload(ctx, path: str, album: str, id: bool) -> None:
    """ 画像をアップロードする
    """
    logger = create_logger()

    service = google_oauth2.get_authorized_service(
        ctx.obj['client'],
        ctx.obj['credential'],
        logger)
    client = google_photos.GooglePhots(service)

    # アルバムに追加する場合は、アルバムIDを探索する。
    # ただし、追加可能なアルバムは一つとするため、見つからない、または複数見つかる場合は、エラーとする。
    album_id = ''
    if album is not None:
        target = 'id' if id else 'title'
        album_list = client.get_album_list()
        albums = [item for item in album_list['albums']
                  if item[target] == album]
        if len(albums) == 0:
            logger.error(f"cannot find album: {album}, id flag = {id}")
            return
        if len(albums) > 1:
            logger.error(
                f"find multiple albums, please identify only one album. album = {album}, id flag = {id}")
            map(lambda x: logger.error(f"album: {x}"), albums)
            return
        album_id = albums[0]['id']

    # 画像のアップロード
    response = client.upload_image(path, album_id)
    logger.info(response)
    logger.info(response['newMediaItemResults'][0]['status'])


if __name__ == "__main__":
    cli.add_command(upload)
    cli(obj={})
