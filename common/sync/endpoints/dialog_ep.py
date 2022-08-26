import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class Dialog:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('Dialog', config)
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

    def start_chat(self, agent_id, data):
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/start_chat"
        response = requests.post(url_request, json=data)
        if response.status_code != 200:
            # messages = [{'type': 'text', 'value': 'sorry ! channel error'}]
            return None
        else:
            # messages = data_json['data']
            data_json = response.json()
            return data_json
        # return message

    def reset_memory(self, agent_id, data):
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/reset_memory"
        response = requests.post(url_request, json=data)
        return response.json()

    def get_memory(self, agent_id, version_id, user_id):
        params = {'version_id': version_id, 'user_id': user_id}
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/get_memory"
        response = requests.get(url_request, params=params)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def save_memory(self, agent_id, data):
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/save_memory"
        response = requests.post(url_request, json=data)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def infer(self, agent_id, data):
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/infer"
        response = requests.post(url_request, json=data)
        if not response:
            return None, -1, "cannot connect to dialog"
        data = response.json().get('data')
        error_code = response.json().get('error_code')
        message = response.json().get('message')

        return data, error_code, message

    def reset_all_dialog(self, agent_id, version_id=None):
        params = {'version_id': version_id}
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/reset_dialog"
        response = requests.get(url_request, params=params)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def reset_specific_dialog(self, agent_id, version_id=None, user_id=None):
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/reset_dialog"
        response = requests.post(
            url_request, json={'user_id': user_id, 'version_id': version_id})
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def update_all_dialog(self, agent_id, version_id=None):
        params = {'version_id': version_id}
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/update_dialog"
        response = requests.get(url_request, params=params)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def update_specific_dialog(self, agent_id, version_id=None, user_id=None):
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/update_dialog"
        response = requests.post(
            url_request, json={'user_id': user_id, 'version_id': version_id})
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def get_status_delay_action(
            self, agent_id, user_id, action_id, version_id=None, message=None
    ):
        json_data = {
            'user_id': user_id,
            'action_id': action_id,
            'message': message
        }
        if version_id:
            json_data['version_id'] = version_id
        url_request = f"{self.service_ep}/internal_api/v1/dialog/agent/{agent_id}/check_delay"
        response = requests.post(url_request, json=json_data)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json
