import logging
import threading

from datetime import timedelta
from minio import Minio
from minio.error import ResponseError, NoSuchBucket, NoSuchKey

from common.utils import config_utils

__logger__ = logging.getLogger()


class MinIO:
    def __init__(self, name=None, app=None, config=None, bucket_name=None):
        self.name = name
        self.bucket_name = bucket_name
        self.region = 'ap-southeast-1'
        self.server = None
        self.access_key = None
        self.secret_key = None
        self.presigned_prefix = None
        self.secure = False
        self.enabled = False
        self.minio_client = None
        self.lock = threading.Lock()

        if app is not None:
            self.init_app(app)

        if config is not None:
            self.init_config(config)

    def is_enabled(self):
        return self.enabled

    def init_config(self, config, name=None):
        if name:
            self.name = name
        service_config = config_utils.get_service_config(
            'MinIO', config, name=self.name)
        if service_config:
            self.enabled = True
            server = service_config.get('SERVER', None)
            if not server: 
                server = service_config.get('server', None)
                
            if server:
                self.server = server
            else:
                host = service_config.get('host', 'localhost')
                port = service_config.get('port', 9000)
                self.server = f'{host}:{port}'
            self.bucket_name = service_config.get(
                'bucket_name', self.bucket_name)
            self.region = service_config.get('region', self.region)
            self.access_key = service_config.get('access_key', None)
            self.secret_key = service_config.get('secret_key', None)
            self.presigned_prefix = service_config.get(
                'presigned_prefix', None)
            __logger__.debug(f'bucket_name = {self.bucket_name}')
            # __logger__.debug(f'access_key = {self.access_key}')
            # __logger__.debug(f'secret_key = {self.secret_key}')
            __logger__.debug(f'server = {self.server}')

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def get_client(self):
        if self.minio_client:
            return self.minio_client
        else:
            with self.lock:
                if self.access_key and self.secret_key:
                    __logger__.debug(f'create minio client')
                    self.minio_client = Minio(self.server, access_key=self.access_key, secret_key=self.secret_key,
                                              secure=self.secure)
                else:
                    self.minio_client = Minio(self.server, secure=self.secure)
                return self.minio_client

    def fput_object(self, file_path, content_type=None, bucket_name=None, object_name=None, metadata=None,
                    progress=None,
                    part_size=None):
        if not self.enabled:
            __logger__.debug(" [minio] Minio is not enabled!")
            return False
        if not bucket_name:
            bucket_name = self.bucket_name

        if object_name is None:
            object_name = file_path

        if object_name[0] == '/':
            object_name = object_name[1:]
        __logger__.debug(f'bucket_name = {self.bucket_name}')
        __logger__.debug(f'object_name = {object_name}')
        __logger__.debug(f'file_path = {file_path}')
        try:
            self.get_client().fput_object(bucket_name, object_name, file_path)
            return True
        except (ResponseError, NoSuchBucket) as err:
            __logger__.debug(err)
            return False

    def fget_object(self, file_path, bucket_name=None, object_name=None, request_headers=None):
        if not self.enabled:
            __logger__.debug(" [minio] Minio is not enabled!")
            return None
        if not bucket_name:
            bucket_name = self.bucket_name

        if object_name is None:
            object_name = file_path

        if object_name[0] == '/':
            object_name = object_name[1:]
        try:
            return self.get_client().fget_object(bucket_name, object_name, file_path)
        except (ResponseError, NoSuchBucket, NoSuchKey) as err:
            __logger__.debug(err)
            return None

    def copy_object(self, object_name, object_source, bucket_source=None, bucket_target=None, copy_conditions=None, metadata=None):
        if not self.enabled:
            __logger__.debug(" [minio] Minio is not enabled!")
            return False
        if object_name is None or object_source is None:
            return False
        if object_name[0] == '/':
            object_name = object_name[1:]

        if object_source[0] == '/':
            object_source = object_source[1:]
        if not bucket_source:
            bucket_source = self.bucket_name
        if not bucket_target:
            bucket_target = self.bucket_name
        try:
            copy_result = self.get_client().copy_object(bucket_target, object_name,
                                                        bucket_source + "/" + object_source,
                                                        copy_conditions, metadata=metadata)
            __logger__.debug(copy_result)
        except ResponseError as err:
            __logger__.debug(err)

    def get_object(self, object_name=None, bucket_name=None, request_headers=None):
        if not self.enabled:
            __logger__.debug(" [minio] Minio is not enabled!")
            return None
        if not bucket_name:
            bucket_name = self.bucket_name

        if object_name is None:
            return None

        if object_name[0] == '/':
            object_name = object_name[1:]
        try:
            data = self.get_client().get_object(bucket_name, object_name)
            for d in data.stream(32*1024):
                yield d
        except (ResponseError, NoSuchBucket, NoSuchKey) as err:
            __logger__.debug(err)
            return None

    def get_object_url(self, object_name, bucket_name=None, expires=timedelta(days=1)):
        if not self.enabled:
            __logger__.debug(" [minio] Minio is not enabled!")
            return None
        if not bucket_name:
            bucket_name = self.bucket_name

        if object_name is None:
            return None

        if object_name[0] == '/':
            object_name = object_name[1:]
        try:
            presigned_url = self.get_client().presigned_get_object(
                bucket_name, object_name, expires=expires)
            if not presigned_url:
                return None

            if self.presigned_prefix:
                if self.server == "s3.amazonaws.com":
                    local_prefix = f"http://{self.bucket_name}.s3-{self.region}.amazonaws.com"
                else:
                    local_prefix = f"http://{self.server}/{self.bucket_name}"

                # __logger__.debug(f"local_prefix: {local_prefix}")
                # __logger__.debug(f"self.presigned_prefix: {self.presigned_prefix}")
                # __logger__.debug(f"presigned_url: {presigned_url}")
                presigned_url = presigned_url.replace(
                    local_prefix, self.presigned_prefix)

            return presigned_url

        except (ResponseError, NoSuchBucket, NoSuchKey) as err:
            __logger__.debug(err)
            return None

    def check_exists(self, object_name=None, bucket_name=None):
        if not self.enabled:
            __logger__.debug(" [minio] Minio is not enabled!")
            return False
        if not bucket_name:
            bucket_name = self.bucket_name

        if object_name is None:
            return False

        if object_name[0] == '/':
            object_name = object_name[1:]
        try:
            __logger__.debug(self.get_client().stat_object(
                bucket_name, object_name))
        except (ResponseError, NoSuchBucket, NoSuchKey) as err:
            __logger__.debug(err)
            return False
        return True
