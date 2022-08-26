import logging
import os
import sys


def init_logger(log_level=logging.DEBUG):
    logger = logging.getLogger()
    wsgi_env = (os.getenv('WSGI_ENV') or 'flask').upper()
    if wsgi_env == 'GUNICORN':
        # forward log to gunicorn
        gunicorn_logger = logging.getLogger('gunicorn.error')

        # formatter = logging.Formatter(
        #     fmt='[%(asctime)s] - [%(process)d] - [%(name)s] - [%(levelname)s] - %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S %z'
        # )
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        # handler.setFormatter(formatter)

        # gunicorn_logger.addHandler(handler)

        logger.handlers = gunicorn_logger.handlers
        logger.setLevel(gunicorn_logger.level)
        # (handler.setFormatter(formatter) for handler in logger.handlers)
    else:
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        # formatter = logging.Formatter(
        #     fmt='[%(asctime)s] - [%(process)d] - [%(name)s] - [%(levelname)s] - %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S %z'
        # )
        # handler.setFormatter(formatter)
        logger.addHandler(handler)

    ignoring_log_level = logging.WARNING
    if log_level > ignoring_log_level:
        ignoring_log_level = log_level

    logging.getLogger('pika').setLevel(ignoring_log_level)
    logging.getLogger('s3transfer').setLevel(ignoring_log_level)
    logging.getLogger('urllib3').setLevel(ignoring_log_level)
    logging.getLogger('botocore').setLevel(ignoring_log_level)

    return logger
