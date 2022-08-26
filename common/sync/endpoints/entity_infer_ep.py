import logging

import requests

from common.utils import config_utils

__logger__ = logging.getLogger()


class EntityInfer:

    def __init__(self, app=None, config=None):
        self.service_ep = "http://localhost:80/"

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('Entity_Infer', config)
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

    def infer_entity(self, agent_id, sample):
        response = requests.post(self.service_ep + '/internal_api/v1/entity_infer', json={'agent_id': agent_id, 'sample': sample})
        if response.status_code == 200:
            entity_infer = response.json()
        else:
            entity_infer = {
                'error_code': response.status_code,
                'message': 'Fail get status intent model entity'
            }
        __logger__.debug(str(entity_infer))
        return entity_infer
