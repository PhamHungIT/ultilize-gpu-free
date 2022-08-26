import logging

import boto3
from botocore.exceptions import ClientError

from common.utils import config_utils

__logger__ = logging.getLogger()


class SES:
    def __init__(self):
        self.aws_region = "ap-southeast-2"
        self.charset = "UTF-8"
        self.client = None
        self.sender = "Vinbase<vinbase.platform@gmail.com>"
        self.app = None
        self.template = None
        self.recipients = None
        self.subject = None
        self.aws_access_key_id = None
        self.enabled = False
        self.aws_secret_access_key = None

    def init_app(self, app, sender):
        # __logger__.debug(f"init  app : {str(app.config)}")
        service_config = config_utils.get_service_config('SES', app.config)
        if service_config:
            self.enabled = True
            self.aws_access_key_id = service_config.get('aws_access_key_id', self.aws_access_key_id)
            self.aws_secret_access_key = service_config.get('aws_secret_access_key', self.aws_secret_access_key)
            self.aws_region = service_config.get('aws_region', self.aws_region)
        self.sender = f'{sender[0]}<{sender[1]}>'
        # __logger__.debug(
        #     f"init  app : {self.aws_access_key_id}, {self.aws_secret_access_key}, {self.aws_region},  {sender}")
        if self.aws_access_key_id and self.aws_secret_access_key:
            self.client = boto3.client('ses', region_name=self.aws_region, aws_access_key_id=self.aws_access_key_id,
                                       aws_secret_access_key=self.aws_secret_access_key)
        else:
            self.client = boto3.client('ses', region_name=self.aws_region)

    def setup(self, template, recipients, subject):
        __logger__.debug(f"setup  template : {template} , recipients : {recipients}, subject: {subject} ")
        self.template = template
        self.recipients = recipients
        self.subject = subject

    def is_enable(self):
        return self.enabled

    def send(self):
        try:
            # Provide the contents of the email.
            response = self.client.send_email(

                Destination={
                    'ToAddresses': [
                        self.recipients,
                    ],
                },
                Message={
                    'Body': {
                        'Html': {
                            'Charset': self.charset,
                            'Data': self.template,
                        },
                        'Text': {
                            'Charset': self.charset,
                            'Data': '',
                        },
                    },
                    'Subject': {
                        'Charset': self.charset,
                        'Data': self.subject,
                    },
                },
                Source=self.sender,
                # If you are not using a configuration set, comment or delete the
                # following line
                # ConfigurationSetName=CONFIGURATION_SET,
            )
        # Display an error if something goes wrong.
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            print("Email sent! Message ID:"),
            print(response['MessageId'])
