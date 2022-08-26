import logging

from common.utils import config_utils

__logger__ = logging.getLogger()


class RemoteFile:
    def __init__(self, name=None, app=None, config=None):
        self.name = name
        self.enabled = False

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def is_enabled(self):
        return self.enabled

    def get_service_config(self, config):
        cfg_key = 'RemoteFile'
        py_key = cfg_key.upper()
        if self.name:
            cfg_key = f'{cfg_key}@{self.name}'
            py_key = f'{py_key}_{self.name.upper()}'

        service_config = None
        if cfg_key in config:
            service_config = config[cfg_key]
        elif py_key in config:
            service_config = config[py_key]
        return service_config

    def init_config(self, config, name=None):
        if name:
            self.name = name
        service_config = config_utils.get_service_config(
            'RemoteFile', config, name=self.name)
        if service_config:
            self.enabled = True

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def upload_file(self):
        pass

    def download_file(self):
        pass

    def put_file(self):
        pass

    def get_file(self):
        pass

    def cp_file(self):
        pass
