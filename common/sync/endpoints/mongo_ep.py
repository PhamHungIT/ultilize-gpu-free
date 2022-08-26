import logging
import urllib

from common.utils import config_utils

__logger__ = logging.getLogger()


class MongoEP:

    def __init__(self, app=None, config=None):
        self.mongo_uri = "mongodb://localhost:27017/"
        self.host = 'localhost'
        self.port = 27017
        self.user = 'slp'
        self.password = ''

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def init_config(self, config):
        service_config = config_utils.get_service_config('MongoDB', config)
        if service_config:
            mongo_uri = service_config.get('mongo_uri', None)
            if mongo_uri:
                self.mongo_uri = mongo_uri
            else:
                self.host = service_config.get('host', self.host)
                self.port = service_config.get('port', self.port)
                self.user = service_config.get('user', self.user)
                self.password = service_config.get('password', self.password)
                self.mongo_uri = 'mongodb://{user}:{password}@{host}:{port}/'.format(user=self.user,
                                                                                     password=urllib.parse.quote_plus(
                                                                                         self.password),
                                                                                     host=self.host,
                                                                                     port=self.port)

    def init_app(self, app):
        if app.config:
            self.init_config(app.config)
