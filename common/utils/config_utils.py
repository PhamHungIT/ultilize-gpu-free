def get_service_config(key, config, name=None):
    cfg_key = key
    py_key = cfg_key.upper()

    if name:
        cfg_key = f'{cfg_key}@{name}'
        py_key = f'{py_key}_{name.upper()}'

    service_config = None
    if cfg_key in config:
        service_config = config[cfg_key]
    elif py_key in config:
        service_config = config[py_key]
    return service_config


def getbool(input):
    return input.lower() in ['true', '1', 't', 'y', 'yes']
