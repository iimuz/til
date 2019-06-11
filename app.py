import logging

import google_oauth2
import google_photos


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


def run() -> None:
    logger = create_logger()

    service = google_oauth2.get_authorized_service()
    client = google_photos.GooglePhots(service)

    album_list = client.get_album_list()
    logger.info(f"album list: {album_list}")

    album_name = "test"
    client.create_new_album(album_name)
    logger.info(f"create new album: {album_name}")

    IMAGE_PATH = '_test/enable_photos_api.png'
    response = client.upload_image(IMAGE_PATH)
    logger.info(response)
    logger.info(response['newMediaItemResults'][0]['status'])


if __name__ == "__main__":
    run()
