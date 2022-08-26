import logging
import threading
import time

import redis

from common.sync.endpoints.constants import BROKER_REDIS
from common.utils import config_utils
from common.utils.exeptions import BreakLoop

__logger__ = logging.getLogger()


class Redis:
    def __init__(self, name=None, app=None, config=None):
        self.name = name
        self.prefix = ''
        self.host = 'localhost'
        self.port = 6379
        self.password = ''
        self.lock = threading.Lock()
        self.r = None
        self.pubsub = None
        self.enabled = False

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def is_enabled(self):
        return self.enabled

    def get_broker_type(self):
        return BROKER_REDIS

    def init_config(self, config, name=None):
        if name:
            self.name = name
        service_config = config_utils.get_service_config(
            'Redis', config, name=self.name)
        if service_config:
            self.enabled = True
            self.prefix = service_config.get('prefix', self.prefix)
            self.host = service_config.get('host', self.host)
            self.port = service_config.get('port', self.port)
            self.password = service_config.get('password', self.password)

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def is_pubsub_open(self):
        try:
            self.pubsub.ping()
            __logger__.debug('Successfully connected to redis pubsub')
            return True
        except redis.exceptions.ConnectionError as r_con_error:
            __logger__.debug('Redis pubsub connection error')
            return False

    def get_pubsub(self):
        r = self.get_client()
        if not self.is_open():
            return None
        else:
            if self.pubsub and self.is_pubsub_open():
                return self.pubsub
            else:
                with self.lock:
                    self.pubsub = r.pubsub()
                    return self.pubsub

    def close_pubsub(self):
        if self.pubsub and self.is_pubsub_open():
            with self.lock:
                self.pubsub.close()

    def is_open(self):
        try:
            self.r.ping()
            return True
        except redis.exceptions.ConnectionError as r_con_error:
            __logger__.debug('Redis connection error')
            return False

    def get_client(self):
        if self.r and self.is_open():
            return self.r
        else:
            with self.lock:
                self.r = redis.StrictRedis(
                    host=self.host, port=self.port, password=self.password, db=0)
                __logger__.debug('Successfully connected to redis')
                return self.r

    def close_client(self):
        if self.r and self.is_open():
            with self.lock:
                self.r.close()

    def ensure_connection_alive(self):
        if not self.enabled:
            __logger__.debug(" [!] Redis is disabled")
            return
        else:
            self.is_open()

    def get(self, key, prefix=None):
        if not prefix:
            prefix = self.prefix
        key = f"{prefix}{key}"
        r = self.get_client()
        return r.get(key)

    def set(self, key, value, prefix=None):
        if not prefix:
            prefix = self.prefix
        key = f"{prefix}{key}"
        r = self.get_client()
        return r.set(key, value)

    def delete(self, key, prefix=None):
        if not prefix:
            prefix = self.prefix
        key = f"{prefix}{key}"
        r = self.get_client()
        return r.delete(key)

    def deleteAll(self, prefix=None):
        if not prefix:
            prefix = self.prefix
        __logger__.debug(f" [!] Deleting all keys with prefix {prefix}")
        pattern = f"{prefix}*"
        r = self.get_client()
        for key in r.scan_iter(pattern):
            r.delete(key)

    def delete_keys(self, keys, prefix=None):
        if not prefix:
            prefix = self.prefix
        prefix = f"{prefix}{keys}"
        __logger__.debug(f" [!] Deleting all keys with prefix {prefix}")
        pattern = f"{prefix}*"
        r = self.get_client()
        for key in r.scan_iter(pattern):
            r.delete(key)

    def exists(self, key, prefix=None):
        if not prefix:
            prefix = self.prefix
        key = f"{prefix}{key}"
        r = self.get_client()
        return r.exists(key) != 0

    def queue_size(self, queue_name, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Return the approximate size of the queue."""
        key = f"{prefix}{queue_name}"
        r = self.get_client()
        return r.llen(key)

    def queue_is_empty(self, queue_name, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Return True if the queue is empty, False otherwise."""
        return self.queue_size(queue_name, prefix) == 0

    def queue_send(self, queue_name, item, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Put item into the queue."""
        key = f"{prefix}{queue_name}"
        r = self.get_client()
        r.lpush(key, item)

    def queue_get(self, queue_name, block=True, timeout=None, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Remove and return an item from the queue. 

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        key = f"{prefix}{queue_name}"
        r = self.get_client()

        if block:
            item = r.brpop(key, timeout=timeout)
        else:
            item = r.rpop(key)

        if item:
            item = item[1]
        return item

    def queue_get_nowait(self, queue_name, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Equivalent to get(False)."""
        return self.queue_get(queue_name, block=False, prefix=prefix)

    def queue_wait_receive(self, queue_name, callback, timeout=None, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Blocking wait the queue to have item, this implementation does NOT follow reliable queue pattern. See more: BRPOPLPUSH"""
        if not self.enabled:
            __logger__.debug(" [!] Redis is disabled")
            return
        else:
            key = f"{prefix}{queue_name}"
            connect_failed_time = time.time()
            retry_delay = 1
            while(True):
                try:
                    r = self.get_client()
                    try:
                        __logger__.debug(
                            ' [*] Waiting for messages. To exit press CTRL+C')
                        while True:
                            item = r.brpop(key, timeout=timeout)
                            retry_delay = 1
                            if not item:
                                __logger__.debug(
                                    f"Timeout after {timeout} seconds ...")
                                raise BreakLoop
                            callback(item[1])
                    except (KeyboardInterrupt, BreakLoop):
                        self.close_client()
                        break
                except Exception as ex:
                    __logger__.debug(f"Error: {ex}")

                    if retry_delay*2 < 60:
                        retry_delay = retry_delay*2
                        connect_failed_time = time.time()
                    else:
                        retry_delay = 60
                        connect_failed_now = time.time()
                        if (connect_failed_now - connect_failed_time) > self.connection_timeout:
                            break

                    __logger__.debug(
                        f"Connection error, retrying after {retry_delay} seconds ...")
                    time.sleep(retry_delay)
                    continue

    def pubsub_publish(self, topic_name, item, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Publish item into the topic."""
        key = f"{prefix}{topic_name}"
        r = self.get_client()
        r.publish(key, item)

    def pubsub_subscribe(self, topic_name, callback, timeout=None, prefix=None):
        if not prefix:
            prefix = self.prefix
        """Blocking wait the queue to have item, this implementation does NOT follow reliable queue pattern. See more: BRPOPLPUSH"""
        if not self.enabled:
            __logger__.debug(" [!] Redis is disabled")
            return
        else:
            key = f"{prefix}{topic_name}"
            connect_failed_time = time.time()
            retry_delay = 1
            while(True):
                try:
                    p = self.get_pubsub()
                    p.subscribe(key)
                    try:
                        __logger__.debug(
                            ' [*] Waiting for messages. To exit press CTRL+C')
                        has_item = True
                        has_item_time = time.time()
                        while True:
                            item = p.get_message()
                            retry_delay = 1
                            if not item:
                                if timeout:
                                    now = time.time()
                                    if (now - has_item_time) > timeout:
                                        __logger__.debug(
                                            f"Timeout after {timeout} seconds ...")
                                        raise BreakLoop
                                if has_item:
                                    __logger__.debug(
                                        f"There is no new event, waiting ...")
                                    has_item = False
                                time.sleep(1)
                                continue

                            has_item = True
                            has_item_time = time.time()
                            __logger__.debug(f"Incoming event ...: {item}")
                            callback(item['data'])
                    except (KeyboardInterrupt, BreakLoop):
                        p.unsubscribe()
                        self.close_pubsub()
                        self.close_client()
                        break
                except Exception as ex:
                    __logger__.debug(f"Error: {ex}")

                    if retry_delay*2 < 60:
                        retry_delay = retry_delay*2
                        connect_failed_time = time.time()
                    else:
                        retry_delay = 60
                        connect_failed_now = time.time()
                        if (connect_failed_now - connect_failed_time) > self.connection_timeout:
                            break

                    __logger__.debug(
                        f"Connection error, retrying after {retry_delay} seconds ...")
                    time.sleep(retry_delay)
                    continue
