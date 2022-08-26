import hashlib
import re
import unidecode


def get_alias(input):
    output = unidecode.unidecode(input)
    output = re.sub(r'\s+', '-', output)
    output = output.lower()
    return output


def validate_special_character(input):
    return re.search('[\[\]`*()/!\\\@#$%^&+={};\':"|,.<>?~]+', input)


def validate_name(name):
    if not name:
        return False
    if name[0] == '_':
        # Name cannot start with a '_'
        return False
    if re.search('[^a-zA-Z0-9\s_-]', name):
        return False
    return True


def get_hash_key(content):
    return hashlib.sha256(content.encode()).hexdigest()


def verify_messages(messages):
    custom_messages = []
    for mes in messages:
        if not (mes.get('type', None) == 'text' and mes.get('value', None) == 'IGNORE_MESSAGE'):
            custom_messages.append(mes)
    return custom_messages


if __name__ == "__main__":
    print(get_alias('đấy là một câu có dấu Lung tung Cực kỳ khó gét ALIÁS'))
