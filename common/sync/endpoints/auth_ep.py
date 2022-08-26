import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class Auth:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80/"
        self.enabled = False

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('Auth', config)
        if service_config:
            self.enabled = True
            server = service_config.get('SERVER', None)
            if not server: 
                server = service_config.get('server', None)
                
            if server:
                self.service_ep = server
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 5016)
                self.service_ep = f'http://{host}:{port}'

    def init_app(self, app):
        if app.config:
            self.init_config(app.config)

    def verify_token(self, token):
        if not self.enabled:
            return 'anonymous'

        if not token:
            return None

        url_request = f"{self.service_ep}/internal_api/v1/authentication/token/verify_token"
        __logger__.debug(f"url_request: {url_request}")
        response = requests.post(url_request, json={'token':token})
        data = response.json()
        __logger__.debug(f"data {data}")
        if data["error_code"] == 0:
            return data.get("data", None)
        else:
            return None
