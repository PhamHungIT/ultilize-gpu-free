import json
from enum import Enum

import requests

__author__ = "tanpk"

URL_BASE = "https://graph.facebook.com/v5.0/"

# send message fields
RECIPIENT_FIELD = "recipient"
TARGET_APP_ID = "target_app_id"
PAGE_INBOX = "263902037430900"
METADATA = "metadata"
METADATA_DEFAULT = "Additional content that the caller wants to set"
TYPE_FIELD = "type"
CONTENT_TYPE_FIELD = "content_type"


class Recipient(Enum):
    PHONE_NUMBER = "phone_number"
    ID = "id"


class Handover(object):
    def __init__(self, access_token):
        self.access_token = access_token

    def pass_thread_control(self, user_id, target_app_id):
        print(f"pass thread control {target_app_id}")
        data = {RECIPIENT_FIELD: self._build_recipient(user_id),
                TARGET_APP_ID: target_app_id,
                METADATA: METADATA_DEFAULT}

        fmt = URL_BASE + "me/pass_thread_control?access_token={token}"
        return requests.post(fmt.format(token=self.access_token),
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(data))

    def take_thread_control(self, user_id):
        data = {RECIPIENT_FIELD: self._build_recipient(user_id),
                METADATA: METADATA_DEFAULT}

        fmt = URL_BASE + "me/take_thread_control?access_token={token}"
        return requests.post(fmt.format(token=self.access_token),
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(data))

    def request_thread_control(self, user_id):
        data = {RECIPIENT_FIELD: self._build_recipient(user_id),
                METADATA: METADATA_DEFAULT}

        fmt = URL_BASE + "me/request_thread_control?access_token={token}"
        return requests.post(fmt.format(token=self.access_token),
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(data))

    @staticmethod
    def _build_recipient(user_id):
        return {Recipient.ID.value: user_id}
