def filter_xss(data, result=None):
    # in case data is dict str
    if isinstance(data, str):
        result = result or check_contain_scripting(data)
    # in case data is dict
    if isinstance(data, dict):
        for key in data:
            children = data.get(key, None)
            if not children:
                continue
            result = result or filter_xss(children)
    # in case data is list
    if isinstance(data, list):
        for children in data:
            if not children:
                continue
            result = result or filter_xss(children)
    # in case data is tuple
    if isinstance(data, tuple):
        for children in data:
            if not children:
                continue
            result = result or filter_xss(children)
    # in case data is set
    if isinstance(data, set):
        for children in data:
            if not children:
                continue
            result = result or filter_xss(children)
    return result


def check_contain_scripting(data):
    if isinstance(data, str):
        if '<script>' in data and '</script>' in data:
            return True
    return False
