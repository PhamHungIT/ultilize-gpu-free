import os
import threading
import configparser
import logging

class Singleton(type):
    _lock = threading.Lock()
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

CONFIG_TYPE_PY = "py"
CONFIG_TYPE_CFG = "cfg"
CONFIG_TYPE_INI = "ini"

class Config(dict):
    def __init__(self, defaults=None):
        dict.__init__(self, defaults or {})
        
    def from_object(self, obj): 
        if isinstance(obj, str):
            return

        for key in dir(obj):
            if key.isupper():
                self[key] = getattr(obj, key)

class ConfigurationManager(object, metaclass=Singleton):
    def __init__(self, config_name=None, config_type=None):
        if not config_type:
            config_type = os.getenv('CONFIG_TYPE') or CONFIG_TYPE_PY

        if config_type not in {CONFIG_TYPE_PY,CONFIG_TYPE_CFG,CONFIG_TYPE_INI}:
            config_type = CONFIG_TYPE_PY

        if config_type == CONFIG_TYPE_PY:
            if not config_name:
                config_name = os.getenv('SERVICE_ENV') or 'dev'
            self.config = Config()
            try:
                from config.config import config_by_name
                self.config.from_object(config_by_name[config_name])
            except ModuleNotFoundError as ex:
                # Error handling
                logger = logging.getLogger("app")
                logger.error(f"*** CANNOT READ CONFIG: {ex}")
                pass
            
        elif config_type == CONFIG_TYPE_CFG or config_type == CONFIG_TYPE_INI:
            if not config_name:
                config_name = os.getenv('SERVICE_CONFIG') or 'config/development.cfg'
            self.config = configparser.ConfigParser()
            self.config.read(config_name)

    def get(self):
        return self.config