import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class Conversation:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80/"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config(
            'Conversation', config)
        if service_config:
            server = service_config.get('SERVER', None)
            if not server:
                server = service_config.get('server', None)

            if server:
                self.service_ep = server
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 5015)
                self.service_ep = f'http://{host}:{port}'

    def init_app(self, app):
        if app.config:
            self.init_config(app.config)

    def update_3rd_party_handover_status(self, user_id, page_id, is_handover):
        url_request = f"{self.service_ep}/internal_api/v1/conversation/agents/conversations/{page_id}/clients/{user_id}/handover"
        __logger__.debug(f"url_request: {url_request}")
        payload = {'is_handover': is_handover}
        response = requests.put(url_request, json=payload)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data = response.json()
            __logger__.debug(f"data {data}")
            if data["error_code"] == 0:
                return data.get("data", None)
            else:
                return None

    def update_handover_status(self, channel_id, user_id, handover_data):
        url_request = f"{self.service_ep}/internal_api/v1/conversation/agents/conversations/{channel_id}/clients/{user_id}/handover"
        __logger__.debug(f"url_request: {url_request}")
        response = requests.put(url_request, json=handover_data)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data = response.json()
            __logger__.debug(f"data {data}")
            if data["error_code"] == 0:
                return data.get("data", None)
            else:
                return None

    def get_client(self, channel_id, user_id):
        if channel_id:
            url_request = f"{self.service_ep}/internal_api/v1/conversation/agents/conversations/{channel_id}/clients/{user_id}"
        else:
            url_request = f"{self.service_ep}/internal_api/v1/conversation/agents/conversations/clients/{user_id}"
        response = requests.get(url_request)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data = response.json()
            __logger__.debug(f"data {data}")
            if data["error_code"] == 0:
                return data.get("data", None)
            else:
                return None
