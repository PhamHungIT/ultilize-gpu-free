import boto3
import botocore
import logging
import threading
from common.utils import config_utils

__logger__ = logging.getLogger()

class EC2:
    def __init__(self, name=None, app=None, config=None):
        self.name = name
        self.ec2 = None
        self.region_name = 'ap-southeast-1'
        self.lock = threading.Lock()
        self.enabled = False

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def is_enabled(self):
        return self.enabled
        
    def init_config(self, config, name=None):
        if name:
            self.name = name
        service_config = config_utils.get_service_config('EC2', config, name=self.name)
        if service_config:
            self.enabled = True
            self.region_name = service_config.get('region_name', self.region_name)

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def get_client(self):
        if self.ec2:
            return self.ec2
        else:
            with self.lock:
                self.ec2 = boto3.client('ec2')
                return self.ec2
