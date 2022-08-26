import logging

import requests

__logger__ = logging.getLogger()

from common.third_party.crawler import Crawler


class FacebookCrawler(Crawler):
    base_endpoint = 'https://graph.facebook.com/v7.0/'
    endpoint_conversation = '{}/conversations?user_id={}&fields=participants,link,updated_time'
    endpoint_conversation_messages = '{}/messages?fields=from,to,created_time,message,attachments,shares,' \
                                     'sticker&limit={}'
    endpoint_message = '{}?fields=message,sticker,created_time,tags,from,attachments,shares,to'
    endpoint_user_info = '{}?fields=first_name,last_name,profile_pic,middle_name'

    def __init__(self, access_token):
        self.access_token = access_token
        super().__init__()

    def build_url(self, endpoint, *params):
        build_url = self.base_endpoint + endpoint.format(*params) + '&access_token={}'.format(self.access_token)
        return build_url

    def get_data(self, endpoint, *params):
        url = self.build_url(endpoint, *params)
        result = requests.get(url).json()
        if 'error' in result:
            return None
        return result

    def get_next(self, next_url):
        result = requests.get(next_url).json()
        return result

    def get_conversation(self, page_id, psid):
        conversation = self.get_data(self.endpoint_conversation, page_id, psid)
        if conversation is None or conversation.get('data', None) is None or len(conversation.get('data')) == 0:
            return None
        return conversation

    def get_messages(self, page_id, psid, limit=25):
        conversation = self.get_conversation(page_id, psid)
        if not conversation:
            return None
        conversation_id = conversation.get('data')[0].get('id')
        messages = self.get_data(self.endpoint_conversation_messages, conversation_id, limit)
        return messages

    def get_conversation_link(self, page_id, psid):
        conversation = self.get_conversation(page_id, psid)
        if not conversation:
            return None
        return conversation.get('data')[0].get('link')

    def get_message_from_id(self, message_id):
        conversation = self.get_data(self.endpoint_message, message_id)
        return conversation

    def get_user_info(self, user_id):
        result = self.get_data(self.endpoint_user_info, user_id)
        __logger__.debug(f'get_user_info = {result}')
        if not result:
            return None, None
        profile_pic = result.get('profile_pic', '')
        first_name = result.get('first_name', '')
        last_name = result.get('last_name', '')
        middle_name = result.get('middle_name', '')
        if middle_name and middle_name != '':
            return profile_pic, last_name + ' ' + middle_name + ' ' + first_name
        else:
            return profile_pic, last_name + ' ' + first_name

    def parse_message_from_3rd(self, message_data):
        need_sync = False
        response_data = []
        response_type = ''
        response_value = ''
        message = message_data.get('message')
        attachments = message_data.get('attachments', None)
        shares = message_data.get('shares', None)
        sticker = message_data.get('sticker', None)

        if None in (attachments, shares) and message != '':
            response_type = 'text'
            response_value = message
            response = {'type': response_type, 'value': response_value}
            response_data.append(response)
        if sticker:
            response_type = 'media'
            response_value = '{"img_url": "' + sticker + \
                             '", "media_type": "image", "title": "", "subtitle": "", "buttons": []}'
            response = {'type': response_type, 'value': response_value}
            response_data.append(response)
        if attachments:
            attachments_data = attachments.get('data')
            for data in attachments_data:
                mime_type = data.get('mime_type')
                if mime_type in ('image/jpeg', 'image/gif'):
                    response_type = 'media'
                    response_value = '{"img_url": "' + data.get('image_data').get(
                        'url') + '", "media_type": "image", "title": "", "subtitle": "", "buttons": []}'
                    need_sync = True
                elif mime_type == 'video/mp4':
                    response_type = 'media'
                    response_value = '{"img_url": "' + data.get('video_data').get(
                        'url') + '", "media_type": "video", "title": "", "subtitle": "", "buttons": []}'
                    need_sync = True
                elif mime_type in ('application/pdf', 'text/csv', 'application/json'):
                    response_type = 'file'
                    response_value = '{"file_url": "' + data.get(
                        "file_url") + '", "file_type": ' + data.get(
                        "mime_type") + '", "name": "' + data.get(
                        "name") + '}'
                response = {'type': response_type, 'value': response_value}
                response_data.append(response)
        # __logger__.debug(f'response_data = {str(response_data)}')
        return response_data, need_sync
