import logging

from flask_mail import Message

from flask_mail import Mail

__logger__ = logging.getLogger()


class Gmail:
    def __init__(self):
        __logger__.debug("init Gmail")
        self.mail = Mail()
        self.sender = ('Vinbase', "vinbase.platform@gmail.com")
        self.app = None
        self.template = None
        self.subject = None
        self.recipients = None

    def init_app(self, app, sender):
        self.mail.init_app(app)
        self.sender = sender
        self.app = app

    def setup(self, template, recipients, subject):
        __logger__.debug(f"setup  template : {template} , recipients : {recipients}, subject: {subject} ")
        self.template = template
        self.recipients = recipients
        self.subject = subject

    def send(self):
        __logger__.debug("send")
        msg = Message(
            self.subject,
            sender=self.sender,
            recipients=[self.recipients])
        msg.html = self.template
        with self.app.app_context():
            self.mail.send(msg)
