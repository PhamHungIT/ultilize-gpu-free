import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class FileService:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config(
            'File_Service', config)
        if service_config:
            server = service_config.get('SERVER', None)
            if not server:
                server = service_config.get('server', None)
            if server:
                self.service_ep = server
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 80)
                self.service_ep = f'http://{host}:{port}'

    def init_app(self, app):
        if app.config:
            self.init_config(app.config)

    def get_file_signature(self, file_id):
        url_request = f"{self.service_ep}/internal_api/v1/file/{file_id}/signature"
        response = requests.get(url_request)
        if response.status_code != 200:
            return None
        else:
            data_json = response.json()
            return data_json

    def create_file(self, data=None):
        url_request = f"{self.service_ep}/internal_api/v1/file/create"
        response = requests.post(url_request, json=data)
        if response.status_code != 200:
            return None
        else:
            data_json = response.json()
            return data_json
