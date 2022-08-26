import boto3
import botocore
import logging
import threading
from common.utils import config_utils
from common.utils.s3_progress import ProgressPercentage

__logger__ = logging.getLogger()

class S3:
    def __init__(self, name=None, app=None, config=None):
        self.name = name
        self.s3 = None
        self.bucket_name = 'bdi'
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
        service_config = config_utils.get_service_config('S3', config, name=self.name)
        if service_config:
            self.enabled = True
            self.bucket_name = service_config.get('bucket_name', self.bucket_name)

    def init_app(self, app, name=None):
        if app.config:
            self.init_config(app.config, name)

    def get_client(self):
        if self.s3:
            return self.s3
        else:
            with self.lock:
                self.s3 = boto3.client('s3')
                return self.s3

    def upload_file(self, file_name, bucket=None, object_name=None):
        if not self.enabled:
            __logger__.debug(" [s3] S3 is not enabled!")
            return False

        if not bucket:
            bucket = self.bucket_name
        
        if object_name is None:
            object_name = file_name
        
        if object_name[0] == '/':
            object_name = object_name[1:]

        s3 = self.get_client()
        try:
            __logger__.debug(f" [s3] uploading ({file_name}) to ({bucket}:{object_name})")
            s3.upload_file(
                file_name, bucket, object_name,
                Callback=ProgressPercentage(file_name)
            )
        except botocore.exceptions.ClientError as e:
            __logger__.error(e)
            return False   
        return True    
    
    def download_file(self, file_name, bucket=None, object_name=None):
        if not self.enabled:
            __logger__.debug(" [s3] S3 is not enabled!")
            return False

        if not bucket:
            bucket = self.bucket_name
        
        if object_name is None:
            object_name = file_name

        if object_name[0] == '/':
            object_name = object_name[1:]

        s3 = self.get_client()
        try:
            __logger__.debug(f" [s3] downloading ({bucket}:{object_name}) to ({file_name})")
            s3.download_file(
                bucket, object_name, file_name
            )
        except botocore.exceptions.ClientError as e:
            __logger__.error(e)
            return False   
        return True 

    def cp_file(self, object_source, object_target, bucket_source = None, bucket_target = None):
        if not self.enabled:
            __logger__.debug(" [s3] S3 is not enabled!")
            return False
            
        if object_source is None or object_target is None:
            return False

        if object_source[0] == '/':
            object_source = object_source[1:]

        if object_target[0] == '/':
            object_target = object_target[1:]

        if not bucket_source:
            bucket_source = self.bucket_name

        if not bucket_target:
            bucket_target = self.bucket_name
        
        s3 = self.get_client()
        try: 
            __logger__.debug(f" [s3] copying ({bucket_source}:{object_source}) to ({bucket_target}:{object_target})")   
            copy_source = {
                'Bucket': bucket_source,
                'Key': object_source
            }
            s3.copy(copy_source, bucket_target, object_target)
        except botocore.exceptions.ClientError as e:
            __logger__.error(e)
            return False   
        return True 
