import google_oauth2
import google_photos


if __name__ == "__main__":
    service = google_oauth2.get_authorized_service()
    client = google_photos.GooglePhots(service)

    album_list = client.get_album_list()
    print(f"album list: {album_list}")

    album_name = "test"
    client.create_new_album(album_name)
    print(f"create new album: {album_name}")

    IMAGE_PATH = '_test/enable_photos_api.png'
    response = client.upload_image(IMAGE_PATH)
    print(response)
    print(response['newMediaItemResults'][0]['status'])
