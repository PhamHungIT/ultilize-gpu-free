import logging

import requests
from common.utils import config_utils

__logger__ = logging.getLogger()


class Passport:
    # Default endpoint
    def __init__(self):
        self.server = "http://localhost:80"
        self.ep_format = "{server}/internal_api/v1/pp"
        self.ep = self.ep_format.format(server=self.server)

    def init_app(self, app):
        self.load_config(app.config)

    def load_config(self, config):
        # __logger__.debug(config)
        service_config = config_utils.get_service_config('Passport', config)
        if service_config:
            server = service_config.get('SERVER', None)
            if not server:
                server = service_config.get('server', None)
            if server:
                self.service_ep = server
                self.ep = self.ep_format.format(server=self.service_ep)
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 80)
                self.service_ep = f'http://{host}:{port}'
                self.ep = self.ep_format.format(server=self.service_ep)

    def get_user_by_token(self, token):
        # logger.debug(self.ep)
        sub_path = "/auth/token"
        response = requests.post(self.ep + sub_path, json={"token": token})
        # logger.debug(response)
        data = response.json()
        if not data:
            return None, "Passport service is not responding"
        user = data.get("data", None)
        msg = data.get("message", None)
        code = data.get("error_code", None)
        return user, code, msg

    def get_user_by_ids(self, ids):
        sub_path = "/users/ids"
        __logger__.debug("get_user_by_ids path " + self.ep + sub_path)
        response = requests.post(self.ep + sub_path, json={"ids": ids})
        data = response.json()
        if not data:
            return None, "Passport service is not responding"
        users = data.get("data", None)
        msg = data.get("message", None)
        code = data.get("error_code", None)
        return users, code, msg

    def get_user_by_email(self, email):
        sub_path = "/users/email"
        __logger__.debug("get_user_by_email path  " + self.ep + sub_path)
        response = requests.post(self.ep + sub_path, json={"email": email})
        data = response.json()
        if not data:
            return None, "Passport service is not responding"
        users = data.get("data", None)
        msg = data.get("message", None)
        code = data.get("error_code", None)
        return users, code, msg
