import os
import logging

from common.sync.endpoints.constants import EV_MODE_PUBSUB, EV_MODE_QUEUE
from common.utils import config_utils

__logger__ = logging.getLogger('event')


class Event:
    def __init__(self, name=None, app=None, config=None, broker=None):
        self.name = name
        self.broker = broker
        self.enabled = False
        self.event_name = None
        self.event_mode = None
        self.event_timeout = 0

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def is_enabled(self):
        return self.enabled

    def init_config(self, config, name=None, broker=None):
        if name:
            self.name = name
        if broker:
            self.broker = broker
        service_config = config_utils.get_service_config(
            'Event', config, name=self.name)
        if service_config:
            disable_event = os.environ.get('DISABLE_EVENT', 'false').lower()
            if disable_event == 'true':
                self.enabled = False
            else:
                self.enabled = True
            self.event_name = service_config.get('event_name', self.event_name)
            self.event_mode = service_config.get('event_mode', self.event_mode)
            self.event_timeout = int(service_config.get('event_timeout', self.event_timeout))

    def init_app(self, app, name=None, broker=None):
        if app.config:
            self.init_config(app.config, name, broker)

    def send(self, message):
        if not self.enabled:
            __logger__.debug(" [ev] Event is disabled!")
            return

        if not self.broker:
            __logger__.debug(" [ev] Broker is not found!")
            return

        if EV_MODE_QUEUE == self.event_mode:
            __logger__.debug(f" [ev] Sending via queue: {self.event_name}, {message}")
            self.broker.queue_send(self.event_name, message)
        elif EV_MODE_PUBSUB == self.event_mode:
            __logger__.debug(f" [ev] Sending via pubsub: {self.event_name}, {message}")
            self.broker.pubsub_publish(self.event_name, message)
        else:
            __logger__.debug(f" [ev] Unsupported event mode {self.event_mode}")

    def wait_receive(self, callback, timeout=None):
        if not self.enabled:
            __logger__.debug(" [ev] Event is disabled!")
            return

        if not self.broker:
            __logger__.debug(" [ev] Broker is not found!")
            return

        if EV_MODE_QUEUE == self.event_mode:
            self.broker.queue_wait_receive(self.event_name, callback, timeout=self.event_timeout)
        elif EV_MODE_PUBSUB == self.event_mode:
            self.broker.pubsub_subscribe(self.event_name, callback, timeout=self.event_timeout)
        else:
            __logger__.debug(f" [ev] Unsupported event mode {self.event_mode}")
