class Crawler():
    def __init__(self):
        pass

    def get_message_from_id(self, message_id):
        return None

    def parse_message_from_3rd(self, message_data):
        return None

    @staticmethod
    def create_crawler(channel_type, data):
        if not channel_type:
            return None
        if channel_type == 'facebook':
            from common.third_party.crawler.fb_crawler_service import FacebookCrawler
            return FacebookCrawler(data)
        if channel_type == 'chatbox':
            from common.third_party.crawler.chatbox_crawler_service import ChatboxCrawler
            return ChatboxCrawler(data)
