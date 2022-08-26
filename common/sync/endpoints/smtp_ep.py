import logging

import smtplib
from email.header import Header
from email.mime.text import MIMEText

__logger__ = logging.getLogger()

from common.utils import config_utils

LOCAL_MAIL_SERVER = 'smtp.gmail.com'
LOCAL_MAIL_USER = 'vinbase.platform@gmail.com'


class SMTP:
    def __init__(self):
        __logger__.debug("init Gmail")
        self.mail_server = LOCAL_MAIL_SERVER
        self.mail_port = 465
        self.message = None
        self.mail_user = LOCAL_MAIL_USER
        self.mail_password = 'vinbase@123456'
        self.sent_from = 'vinbase.platform@gmail.com'
        self.sent_to = []

    def init_config(self, config):
        # __logger__.debug(f"init  app : {str(config)}")
        mail_config = config_utils.get_service_config('Mail', config)
        if mail_config:
            self.mail_server = mail_config.get('mail_server')
            self.mail_port = mail_config.get('mail_port')
            self.mail_user = mail_config.get('mail_user')
            self.mail_password = mail_config.get('mail_password')
            self.sent_from = mail_config.get('sent_from', self.sent_from)

    def setup(self, template, sent_to, subject):
        __logger__.debug(f"setup  template : {template} , sent_to : {sent_to}, subject: {subject} ")
        # __logger__.debug(
        #     f"mail_server : {self.mail_server} , mail_port : {self.mail_port}, mail_user: {self.mail_user}, "
        #     f"mail_password: {self.mail_password}  ")
        self.sent_to = sent_to
        self.message = MIMEText(str(template), 'html', _charset="UTF-8")
        self.message.set_charset('utf8')
        self.message['Subject'] = Header(subject.encode('utf-8'), "UTF-8").encode()
        if self.mail_server == LOCAL_MAIL_SERVER:
            self.message['From'] = f'Vinbase.ai {self.sent_from}'
        else:
            self.message['From'] = f'Vinbase.ai <{self.sent_from}>'
        self.message['To'] = ','.join(sent_to)

    def send(self):
        try:
            if self.mail_server != LOCAL_MAIL_SERVER:
                server = smtplib.SMTP(self.mail_server, self.mail_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.mail_server, self.mail_port)
            server.ehlo()
            server.login(self.mail_user, self.mail_password)
            server.sendmail(self.sent_from, self.sent_to, self.message.as_string())
            server.close()
            __logger__.debug('Email sent!')
        except Exception as ex:
            __logger__.debug(f'Something went wrong... {ex}')
