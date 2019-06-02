import pathlib

import requests


class GooglePhots:
    """ Google photosへの処理を実行する
    """
    def __init__(self, service):
        self._service = service
        pass

    def create_new_album(self, album_name: str):
        """ 新規にアルバムを生成する

        Note
        ---
        同一のアルバム名が存在しても、同じ名称でアルバムを生成する。
        (=同一の名称となるアルバムが複数存在できる)
        """
        new_album = {'album': {'title': album_name}}
        response = self._service.albums().create(body=new_album).execute()
        return response['id']

    def get_album_list(self):
        """ 既存のアルバムリストを取得する
        """
        return self._service.albums().list().execute()

    def upload_image(self, image_path: str):
        """ 画像をアップロードする
        """
        with open(image_path, 'rb') as image_data:
            URL = 'https://photoslibrary.googleapis.com/v1/uploads'
            HEADERS = {
                'Authorization': "Bearer " + self._service._http.request.credentials.access_token,
                'Content-Type': 'application/octet-stream',
                'X-Goog-Upload-File-Name': pathlib.Path(image_path).name,
                'X-Goog-Upload-Protocol': "raw",
            }
            response = requests.post(URL, data=image_data, headers=HEADERS)

        upload_token = response.content.decode('utf-8')
        new_item = {'newMediaItems': [{'simpleMediaItem': {'uploadToken': upload_token}}]}
        response = self._service.mediaItems().batchCreate(body=new_item).execute()

        return response
