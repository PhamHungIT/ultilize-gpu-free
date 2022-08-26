import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class Sender:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80/"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('Sender', config)
        if service_config:
            server = service_config.get('SERVER', None)
            if not server: 
                server = service_config.get('server', None)
                
            if server:
                self.service_ep = server
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 5017)
                self.service_ep = f'http://{host}:{port}'

    def init_app(self, app):
        if app.config:
            self.init_config(app.config)

    def send(self, request_body):
        url_request = f"{self.service_ep}/internal_api/v1/sender/send"
        response = requests.post(url_request, json=request_body)
        data = response.json()
        return data

    def sync_message(self, request_body):
        url_request = f"{self.service_ep}/internal_api/v1/sender/sync_message"
        response = requests.post(url_request, json=request_body)
        data = response.json()
        return data
