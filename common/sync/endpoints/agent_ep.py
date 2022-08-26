import logging

import requests
import json
from common.utils import config_utils
from common.utils.constants import (TRAINING_STATUS_DONE, TRAINING_STATUS_STOPPED,
                                    TRAINING_STATUS_STOPPING, TRAINING_STATUS_TRAINING)

__logger__ = logging.getLogger()


class Agent:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80/"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('Agent', config)
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

    def get_agent_info_by_id(self, agent_id):
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/{agent_id}"  # noqa
        response = requests.get(url_request)
        data = response.json()
        if data["error_code"] == 0:
            return data.get("data", None)
        else:
            return None

    def get_token_by_id(self, token_id):
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/channels/token/{token_id}"
        # __logger__.debug(f"url_request: {url_request}")
        response = requests.get(url_request)
        data = response.json()
        # __logger__.debug(f"data {data}")
        if data["error_code"] == 0:
            return data.get("data", None)
        else:
            return None

    def get_all_channels(self):
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/channels"
        response = requests.get(url_request)
        data = response.json()
        # __logger__.debug(f"data {data}")
        if data["error_code"] == 0:
            return data.get("data", None)
        else:
            return None

    def get_channel(self, channel_id):
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/channels/{channel_id}"
        response = requests.get(url_request)
        data = response.json()
        # __logger__.debug(f"data {data}")
        if data["error_code"] == 0:
            return data.get("data", None)
        else:
            return None

    def get_skill_requirement_form(self, agent_id, version_id, skill_id):
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/{agent_id}/versions/{version_id}/skills/{skill_id}/requirements/form"
        response = requests.get(url_request)
        if response.status_code != 200:
            __logger__.info(
                f'{url_request} status code = {response.status_code}')
            return None
        else:
            data_json = response.json()
            return data_json

    def post_progress(self, agent_id, progress, progress_type, version_id=None, torch_script_data=None):
        __logger__.debug(f"{agent_id} is training at {progress}")
        if version_id is not None:
            url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/{agent_id}/versions/{version_id}/train/status"
        else:
            url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/{agent_id}/train/status"
        __logger__.debug(f"url_request: {url_request}")
        response = requests.post(
            url_request, json={f'{progress_type}_progress': progress})
        data = response.json()
        __logger__.debug(f"response code: {data['error_code']}")

        if progress >= 1:
            __logger__.debug(f"{agent_id} is training is finished")

            response = requests.post(
                url_request, json={f'training_{progress_type}': False, 'torch_script_data': torch_script_data})
            data = response.json()
            __logger__.debug(f"response code: {data['error_code']}")

    def update_job_progress(self, agent_id, version_id, job_id, progress, torch_script_data=None, status=TRAINING_STATUS_TRAINING):
        data = {'progress': progress, 'status': status}
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/{agent_id}/versions/{version_id}/job/{job_id}/status"
        if status == TRAINING_STATUS_DONE and torch_script_data is not None:
            data['torch_script_data'] = torch_script_data
        response = requests.post(
            url_request, json=data)
        if response.status_code != 200:
            return None
        else:
            return response.json()

    def get_matching_comment_config(self, agent_id, version_id, page_id, post_id):
        url_request = f"{self.service_ep}/internal_api/v1/chatbot/agents/{agent_id}/versions/{version_id}/comments/configs/search"  # noqa
        data = {"page_id": page_id, "post_id": post_id}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        __logger__.debug(f'url_request: {url_request}')
        response = requests.post(
            url_request, data=json.dumps(data), headers=headers)
        __logger__.debug(f'response {response}')
        data = response.json()
        if data["error_code"] == 0:
            return data.get("data", None)
        else:
            return None
