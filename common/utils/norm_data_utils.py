from datetime import timezone


def convert_fields(item):
    _id = item.pop('_id', None)
    if _id:
        item['id'] = str(_id)

    if 'create_time' in item:
        item['create_time'] = int(item['create_time'].replace(tzinfo=timezone.utc).timestamp())

    if 'update_time' in item:
        item['update_time'] = int(item['update_time'].replace(tzinfo=timezone.utc).timestamp())

    if 'last_received' in item:
        item['last_received'] = int(item['last_received'].replace(tzinfo=timezone.utc).timestamp())

    if 'created_time' in item:
        item['created_time'] = int(item['created_time'].replace(tzinfo=timezone.utc).timestamp())

    return item
