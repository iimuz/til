import pathlib
from httplib2 import Http
from typing import List

import googleapiclient.discovery as gdiscovery
from oauth2client import client
from oauth2client.file import Storage


def get_authorized_service() -> gdiscovery.Resource:
    """ Google Photosの認証を実行
    """
    SCOPES = ['https://www.googleapis.com/auth/photoslibrary']
    OAUTH_CLIENT_FILE = './_test/client_id.json'
    CREDENTIALS_FILE = './_test/credentials.json'
    API_SERVICE_NAME = 'photoslibrary'
    API_VERSION = 'v1'

    credentials = None
    if pathlib.Path(CREDENTIALS_FILE).exists():
        storage = Storage(CREDENTIALS_FILE)
        credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = _get_credentials_file(
            OAUTH_CLIENT_FILE,
            SCOPES,
            CREDENTIALS_FILE
        )

    return gdiscovery.build(API_SERVICE_NAME, API_VERSION, http=credentials.authorize(Http()))


def _get_credentials_file(
    client_file_path: str,
    scopes: List[str],
    credentials_file_path: str
) -> client.OAuth2Credentials:
    """ credentialsファイルを生成する
    """
    flow = client.flow_from_clientsecrets(
        client_file_path,
        scope=scopes,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'
    )
    auth_uri = flow.step1_get_authorize_url()

    print(f'Open below URL by browser: {auth_uri}')
    token = input("Input your code: ")

    credentials = flow.step2_exchange(token)
    Storage(credentials_file_path).put(credentials)

    return credentials
