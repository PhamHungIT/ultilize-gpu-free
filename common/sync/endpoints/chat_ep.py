import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class Chat:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80/"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('Chat', config)
        if service_config:
            server = service_config.get('SERVER', None)
            if not server:
                server = service_config.get('server', None)

            if server:
                self.service_ep = server
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 5001)
                self.service_ep = f'http://{host}:{port}'

    def init_app(self, app):
        if app.config:
            self.init_config(app.config)

    def save_memory(self, agent_id, data):
        url_request = f"{self.service_ep}/internal_api/v1/channel_chat/agent/{agent_id}/save_memory"  # noqa
        response = requests.post(url_request, json=data)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def update_handover_status(self, channel_id, user_id, is_handover):
        url_request = f"{self.service_ep}/internal_api/v1/channel_chat/channels/{channel_id}/clients/{user_id}/handover"
        response = requests.post(
            url_request, json={'is_handover': is_handover})
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json
