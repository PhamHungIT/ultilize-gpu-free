from common.third_party.crawler import Crawler


class ChatboxCrawler(Crawler):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def get_user_info(self, user_id):
        user_name = self.data.get('user_name', '')
        profile_pic = self.data.get('profile_pic', '')

        return profile_pic, user_name

    def get_message_from_id(self, message_id):
        return None

    def parse_message_from_3rd(self, message_data):
        return None
