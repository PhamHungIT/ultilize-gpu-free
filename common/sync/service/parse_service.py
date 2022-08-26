import json

from common.utils.profiling import timeit
from common.third_party.pymessager import GenericElement, ActionButton, ButtonType, QuickReply


from common.utils import logger


@timeit
def parse_default_message(messages):
    return messages


@timeit
def parse_facebook_message(messages):
    result = []
    for mes in messages:
        logger.debug("mes {}".format(mes))
        if mes.get('type', None) == 'text':
            result.append(mes)
        elif mes.get('type', None) == 'handover':
            result.append(mes)
        elif mes.get('type', None) == 'media':
            value = json.loads(mes['value'])
            logger.debug("media {}".format(value))
            if value.get('media_type', None) == 'image':
                result.append({
                    "type": "image",
                    "value": value['img_url']
                })
            elif value.get('media_type', None) == 'video':
                media_url = value.get('img_url', None)
                if 'facebook' in media_url:
                    result.append({
                        "type": "video",
                        "value": media_url
                    })

        elif mes.get('type', None) == 'image':
            value = json.loads(mes['value'])
            logger.debug("image {}".format(value))

            result.append({
                "type": "image",
                "value": value['img_url']
            })

        elif mes.get('type', None) == 'button':
            value = json.loads(mes['value'])
            buttons = []
            for but in value['buttons']:
                if but.get('type', None) == 'link':
                    buttons.append(ActionButton(ButtonType.WEB_URL, but.get('title', ""), url=but['value']))
                else:
                    buttons.append(ActionButton(ButtonType.POSTBACK, but.get('title', ""), payload=but['value']))
            result.append({
                "type": "button",
                "title": value.get('title', ""),
                "buttons": buttons
            })

        elif mes.get('type', None) == 'list':
            values = json.loads(mes['value'])
            element = [GenericElement(
                value.get('title', ""),
                value.get('subtitle', ""),
                value.get('img_url', ""), [
                    ActionButton(ButtonType.POSTBACK,
                                 but['title'],
                                 payload=but['value']) for but in value['buttons']]) for value in values]
            result.append({
                "type": "list",
                "value": element
            })
        elif mes.get('type', None) == 'carousel':
            values = json.loads(mes['value'])
            generic_element = []
            for value in values:
                action_buttons = []
                for i in range(min(3, len(value['buttons']))):
                    button = value['buttons'][i]
                    logger.debug(f"^^ {button}")
                    if button['type'] == 'link':
                        action_buttons.append(ActionButton(ButtonType.WEB_URL,
                                                           button['title'],
                                                           url=button['value']))
                    else:
                        action_buttons.append(ActionButton(ButtonType.POSTBACK,
                                                           button['title'],
                                                           payload=button['value']))
                generic_element.append(GenericElement(value.get('title', ""),
                                                      value.get('subtitle', ""),
                                                      value.get('img_url', ""),
                                                      action_buttons))
            element = generic_element
            result.append({
                "type": "carousel",
                "value": element
            })
        elif mes.get('type', None) == 'quick_reply':
            values = json.loads(mes['value'])
            quick_replies = [QuickReply(rep['title'], rep['value']) for rep in values['buttons']]
            text = values['title']
            result.append({
                "type": "quick_reply",
                "quick_replies": quick_replies,
                "text": text
            })
    return result


@timeit
def parse_zalo_message(messages):
    result = [mes.get('value', "") for mes in messages]
    return result


@timeit
def parse_event_facebook(webhook_event):
    message = None
    webhook_data = {
        'type': None,
        'data': None
    }
    if 'message' in webhook_event:
        webhook_data['type'] = 'message'
        if 'quick_reply' in webhook_event['message']:
            message = webhook_event['message']['quick_reply'].get('payload', None)
        elif 'text' in webhook_event['message']:
            message = webhook_event['message']['text']
    elif 'postback' in webhook_event:
        webhook_data['type'] = 'message'
        message = webhook_event['postback'].get('payload', None)
    elif 'request_thread_control' in webhook_event:
        webhook_data = {
            'type': 'request_thread_control',
            'data': webhook_event['request_thread_control'].get('requested_owner_app_id', None)
        }
        return webhook_data
    webhook_data['data'] = message
    return webhook_data
