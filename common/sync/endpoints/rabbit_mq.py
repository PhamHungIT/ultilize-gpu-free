import os
import logging
import threading
import time

import pika

from common.sync.endpoints.constants import BROKER_RABBITMQ
from common.utils import config_utils

__logger__ = logging.getLogger()


class Rabbit:
    def __init__(self, name=None, app=None, config=None):
        self.name = name
        self.host = 'localhost'
        self.port = 5672
        self.user = 'guest'
        self.password = 'guest'
        self.lock = threading.Lock()
        self.connection = None
        self.enabled = False

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def is_enabled(self):
        return self.enabled

    def get_broker_type(self):
        return BROKER_RABBITMQ

    def init_config(self, config, name=None):
        if name:
            self.name = name
        service_config = config_utils.get_service_config(
            'RabbitMQ', config, name=self.name)
        if service_config:
            disable_queue = os.environ.get('DISABLE_QUEUE', 'false').lower()
            if disable_queue == 'true':
                self.enabled = False
            else:
                self.enabled = True
            self.host = service_config.get('host', self.host)
            self.port = service_config.get('port', self.port)
            self.user = service_config.get('user', self.user)
            self.password = service_config.get('password', self.password)

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def get_connection(self):
        if self.connection and self.connection.is_open:
            return self.connection
        else:
            with self.lock:
                credentials = pika.PlainCredentials(self.user, self.password)
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials, heartbeat=3600,
                                              blocked_connection_timeout=86400))
                return self.connection

    def close_connection(self):
        if self.connection and self.connection.is_open:
            with self.lock:
                self.connection.close()

    def ensure_connection_alive(self):
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:
            connection = self.get_connection()
            connection.process_data_events()

    def pubsub_publish(self, to_exchange, message):
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:
            credentials = pika.PlainCredentials(
                        self.user, self.password)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials))
            try:
                channel = connection.channel()
                channel.exchange_declare(
                    exchange=to_exchange, exchange_type='fanout')
                channel.basic_publish(
                    exchange=to_exchange,
                    routing_key='',
                    body=message)

                __logger__.debug(" [x] Sent %r" % message)
                connection.close()
            except KeyboardInterrupt:
                connection.close()

    @staticmethod
    def pubsub_callback_wrapper(callback):
        def func(ch, method, properties, body):
            __logger__.debug(" [x] Received %r" % body)
            callback(body)
            __logger__.debug(" [x] Done")
        return func

    def pubsub_subscribe(self, from_exchange, callback, timeout=None):
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:
            retry_delay = 1
            while(True):
                try:                    
                    connection = self.get_connection()
                    channel = connection.channel()

                    channel.exchange_declare(
                        exchange=from_exchange, exchange_type='fanout')
                    result = channel.queue_declare(queue='', exclusive=True)
                    queue_name = result.method.queue
                    channel.queue_bind(
                        exchange=from_exchange, queue=queue_name)
                    channel.basic_consume(queue=queue_name, on_message_callback=Rabbit.pubsub_callback_wrapper(
                        callback), auto_ack=True)

                    try:
                        __logger__.debug(
                            ' [*] Waiting for messages. To exit press CTRL+C')
                        retry_delay = 1
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                        break
                except pika.exceptions.ConnectionClosedByBroker:
                    # Uncomment this to make the example not attempt recovery
                    # from server-initiated connection closure, including
                    # when the node is stopped cleanly
                    #
                    # break
                    if retry_delay*2 < 60:
                        retry_delay = retry_delay*2
                    else:
                        retry_delay = 60
                    __logger__.debug(
                        "Broker node is stopped, retrying after {} seconds ...".format(retry_delay))
                    time.sleep(retry_delay)
                    continue
                # Do not recover on channel errors
                except pika.exceptions.AMQPChannelError as err:
                    __logger__.debug(
                        "Caught a channel error: {}, stopping...".format(err))
                    break
                # Recover on all other connection errors
                except pika.exceptions.AMQPConnectionError:
                    if retry_delay*2 < 60:
                        retry_delay = retry_delay*2
                    else:
                        retry_delay = 60
                    __logger__.debug(
                        "Connection was closed, retrying after {} seconds ...".format(retry_delay))
                    time.sleep(retry_delay)
                    continue

    def queue_send(self, to_queue, message):
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:            
            credentials = pika.PlainCredentials(
                        self.user, self.password)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials))
            try:
                channel = connection.channel()
                channel.queue_declare(queue=to_queue, durable=True)
                channel.basic_publish(
                    exchange='',
                    routing_key=to_queue,
                    body=message,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    ))

                __logger__.debug(" [x] Sent %r" % message)
                connection.close()
            except KeyboardInterrupt:
                connection.close()

    @staticmethod
    def queue_callback_wrapper(callback):
        def func(ch, method, properties, body):
            __logger__.debug(" [x] Received %r" % body)
            callback(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            __logger__.debug(" [x] Done")
        return func

    def queue_wait_receive(self, from_queue, callback, timeout=None):
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:
            retry_delay = 1
            while(True):
                try:
                    connection = self.get_connection()
                    channel = connection.channel()

                    channel.queue_declare(queue=from_queue, durable=True)

                    channel.basic_qos(prefetch_count=1)
                    channel.basic_consume(
                        from_queue, Rabbit.queue_callback_wrapper(callback))

                    try:
                        __logger__.debug(
                            ' [*] Waiting for messages. To exit press CTRL+C')
                        retry_delay = 1
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        self.close_connection()
                        break
                except pika.exceptions.ConnectionClosedByBroker:
                    # Uncomment this to make the example not attempt recovery
                    # from server-initiated connection closure, including
                    # when the node is stopped cleanly
                    #
                    # break
                    if retry_delay*2 < 60:
                        retry_delay = retry_delay*2
                    else:
                        retry_delay = 60
                    __logger__.debug(
                        "Broker node is stopped, retrying after {} seconds ...".format(retry_delay))
                    time.sleep(retry_delay)
                    continue
                # Do not recover on channel errors
                except pika.exceptions.AMQPChannelError as err:
                    __logger__.debug(
                        "Caught a channel error: {}, stopping...".format(err))
                    break
                # Recover on all other connection errors
                except pika.exceptions.AMQPConnectionError:
                    if retry_delay*2 < 60:
                        retry_delay = retry_delay*2
                    else:
                        retry_delay = 60
                    __logger__.debug(
                        "Connection was closed, retrying after {} seconds ...".format(retry_delay))
                    time.sleep(retry_delay)
                    continue
