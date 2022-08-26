import os
import logging
import threading
import time

import boto3

from common.sync.endpoints.constants import BROKER_SQS
from common.utils import config_utils
from common.utils.exeptions import BreakLoop

__logger__ = logging.getLogger('sqs')


class SQS:
    def is_enabled(self):
        return self.enabled

    def __init__(self, name=None, app=None, config=None):
        self.name = name
        self.wait_time = 0
        self.region_name = 'ap-southeast-1'
        self.connection_timeout = 180
        self.lock = threading.Lock()
        self.sqs = None
        self.enabled = False

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def get_broker_type(self):
        return BROKER_SQS

    def init_config(self, config, name=None):
        if name:
            self.name = name
        service_config = config_utils.get_service_config(
            'SQS', config, name=self.name)
        if service_config:
            __logger__.debug(" [x] SQS is configurated")
            disable_queue = os.environ.get('DISABLE_QUEUE', 'false').lower()
            if disable_queue == 'true':
                self.enabled = False
            else:
                self.enabled = True
            self.wait_time = int(service_config.get('wait_time', self.wait_time))
            self.region_name = service_config.get('region_name', self.region_name)
            self.connection_timeout = int(service_config.get('connection_timeout', self.connection_timeout))

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def get_queue(self):
        if self.sqs:
            return self.sqs
        else:
            with self.lock:
                self.sqs = boto3.client('sqs', self.region_name)
                return self.sqs

    def queue_send(self, to_queue, message):
        __logger__.debug(f" [x] Sending to queue: {to_queue}")
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:
            sqs = self.get_queue()
            try:
                response = sqs.send_message(
                    QueueUrl=to_queue,
                    MessageBody=(message)
                )
                __logger__.debug(" [x] Sent %r" % message)
            except Exception as ex:
                __logger__.debug(f" [x] Send error: {ex}")

    @staticmethod
    def queue_callback_wrapper(callback):
        def func(sqs, from_queue, receipt_handle, body):
            __logger__.debug(" [x] Received %r" % body)
            callback(body)

            sqs.delete_message(
                QueueUrl=from_queue,
                ReceiptHandle=receipt_handle
            )
            __logger__.debug(" [x] Done")
        return func

    def queue_wait_receive(self, from_queue, callback, timeout=None):
        if not self.enabled:
            __logger__.debug(" [!] MQ is disabled")
            return
        else:
            retry_delay = 1
            connect_failed_time = time.time()
            while(True):
                try:
                    sqs = self.get_queue()
                    # Receive message from SQS queue
                    try:
                        has_item = True
                        has_item_time = time.time()
                        __logger__.debug(
                            ' [*] Waiting for messages. To exit press CTRL+C')
                        while True:
                            response = sqs.receive_message(
                                QueueUrl=from_queue,
                                MaxNumberOfMessages=1,
                                WaitTimeSeconds=self.wait_time
                            )
                            retry_delay = 1
                            message = None

                            if response:
                                messages = response.get('Messages', [])
                                if messages:
                                    message = messages[0]

                            if not message:
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
                            __logger__.debug(f"Incoming event ...")
                            receipt_handle = message['ReceiptHandle']
                            wrapped_callback = SQS.queue_callback_wrapper(
                                callback)
                            wrapped_callback(
                                sqs, from_queue, receipt_handle, message['Body'])

                    except (KeyboardInterrupt, BreakLoop):
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
