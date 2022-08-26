#!/usr/bin/python
# -*- coding: utf8 -*-
import json
from enum import Enum
from common.utils import logger
from common.utils.profiling import timeit

import requests

__author__ = "tanpk"

URL_BASE = "https://openapi.zalo.me/v2.0/"

# send message fields
RECIPIENT_FIELD = "recipient"
MESSAGE_FIELD = "message"
ATTACHMENT_FIELD = "attachment"
TYPE_FIELD = "type"
TEMPLATE_TYPE_FIELD = "template_type"
MEDIA_TYPE_FIELD = "media_type"
TEXT_FIELD = "text"
TITLE_FIELD = "title"
URL_FIELD = 'url'
SUBTITLE_FIELD = "subtitle"
IMAGE_FIELD = "image_url"
DEFAULT_ACTION = 'default_action'
BUTTONS_FIELD = "buttons"
PAYLOAD_FIELD = "payload"
URL_FIELD = "url"
ELEMENTS_FIELD = "elements"
QUICK_REPLIES_FIELD = "quick_replies"
CONTENT_TYPE_FIELD = "content_type"
BUTTONS_MAX_LENGTH = 5

# received message fields
POSTBACK_FIELD = "postback"


class Recipient(Enum):
    USER_ID = "user_id"
    ID = "id"


class TemplateType(Enum):
    LIST = "list"
    BUTTON = "button"
    RECEIPT = "receipt"
    MEDIA = 'media'


class MessageType(Enum):
    TEXT = "text"
    ATTACHMENT = "attachment"


class AttachmentType(Enum):
    IMAGE = "image"
    VIDEO = 'video'
    TEMPLATE = "template"


class ActionType(Enum):
    OPEN_URL = 'oa.open.url'
    QUERY_SHOW = 'oa.query.show'
    QUERY_HIDE = 'oa.query.hide'
    OPEN_SMS = 'oa.open.sms'
    OPEN_PHONE = 'oa.open.phone'


class ActionButton:
    def __init__(self, button_type, title, payload=None):
        self.button_type = button_type
        self.title = title
        self.payload = payload
        if self.payload:
            self.payload = payload.to_dict()

    def to_dict(self):
        button_dict = {TYPE_FIELD: self.button_type}
        if self.title:
            button_dict[TITLE_FIELD] = self.title
        if self.payload:
            button_dict[PAYLOAD_FIELD] = self.payload
        return button_dict


class Element:
    def __init__(self, title, subtitle, image_url, default_action=None):
        self.title = title
        self.subtitle = subtitle
        self.image_url = image_url
        self.default_action = default_action

    def to_dict(self):
        element_dict = {}
        if self.title:
            element_dict[TITLE_FIELD] = self.title
        if self.subtitle:
            element_dict[SUBTITLE_FIELD] = self.subtitle
        if self.image_url:
            element_dict[IMAGE_FIELD] = self.image_url
        if self.default_action:
            element_dict[DEFAULT_ACTION] = self.default_action.to_dict()
        return element_dict


class Payload:
    def __init__(self, text_chat=None, content=None, phone_core=None, url=None):
        self.text_chat = text_chat
        self.content = content
        self.phone_core = phone_core
        self.url = url

    def to_dict(self):
        if self.url:
            return {
                'url': self.url
            }
        if self.text_chat:
            return self.text_chat
        if self.phone_core and self.content:
            return {
                "content": self.content,
                "phone_code": self.phone_core
            }
        if self.phone_core:
            return {
                "phone_code": self.phone_core
            }
        return None


class DefaultAction:
    def __init__(self, type_button, url=None, payload=None):
        self.url = url
        self.payload = payload
        self.type_button = type_button
        if self.payload:
            self.payload = payload.to_dict()

    def to_dict(self):
        default_action = {}
        if self.type_button == ActionType.OPEN_URL.value:
            default_action[TYPE_FIELD] = self.type_button
            default_action[URL_FIELD] = self.url
        else:
            default_action[TYPE_FIELD] = self.type_button
            default_action[PAYLOAD_FIELD] = self.payload
        return default_action


class Messager(object):
    def __init__(self, access_token):
        self.access_token = access_token

    @staticmethod
    def _build_recipient(user_id):
        return {Recipient.USER_ID.value: user_id}

    @timeit
    def send_text(self, user_id, text):
        return self._send({RECIPIENT_FIELD: self._build_recipient(user_id),
                           MESSAGE_FIELD: {MessageType.TEXT.value: text}})

    @timeit
    def send_image(self, user_id, url_image, title_text=None):
        return self._send({RECIPIENT_FIELD: self._build_recipient(user_id),
                           MESSAGE_FIELD: {
                               # TEXT_FIELD: title_text,
                               ATTACHMENT_FIELD: {
                                   TYPE_FIELD: AttachmentType.TEMPLATE.value,
                                   PAYLOAD_FIELD: {
                                       TEMPLATE_TYPE_FIELD: TemplateType.MEDIA.value,
                                       ELEMENTS_FIELD: [
                                           {
                                               MEDIA_TYPE_FIELD: AttachmentType.IMAGE.value,
                                               URL_FIELD: url_image
                                           }
                                       ]
                                   }
                               }
                           }})

    @timeit
    def send_lists(self, user_id, element_list, button_list):
        buttons = [button.to_dict() for button in button_list]
        buttons = buttons[0:min(len(buttons), BUTTONS_MAX_LENGTH)]
        elements = [element.to_dict() for element in element_list]
        if len(buttons) == 0:
            return self._send({RECIPIENT_FIELD: self._build_recipient(user_id),
                               MESSAGE_FIELD: {
                                   ATTACHMENT_FIELD: {
                                       TYPE_FIELD: AttachmentType.TEMPLATE.value,
                                       PAYLOAD_FIELD: {
                                           TEMPLATE_TYPE_FIELD: TemplateType.LIST.value,
                                           ELEMENTS_FIELD: elements,
                                       }
                                   }
                               }})

        return self._send({RECIPIENT_FIELD: self._build_recipient(user_id),
                           MESSAGE_FIELD: {
                               ATTACHMENT_FIELD: {
                                   TYPE_FIELD: AttachmentType.TEMPLATE.value,
                                   PAYLOAD_FIELD: {
                                       TEMPLATE_TYPE_FIELD: TemplateType.LIST.value,
                                       ELEMENTS_FIELD: elements,
                                       BUTTONS_FIELD: buttons
                                   }
                               }
                           }})

    @timeit
    def _send(self, message_data):
        post_message_url = URL_BASE + "oa/message?access_token={token}".format(
            token=self.access_token)
        response_message = json.dumps(message_data)
        logger.debug(response_message)
        req = requests.post(post_message_url,
                            headers={"Content-Type": "application/json"},
                            data=response_message)
        logger.debug(
            f"[{req.status_code}/{req.reason}/{req.text}] to {message_data[RECIPIENT_FIELD]} content: {message_data[MESSAGE_FIELD]}")
        return req, message_data
