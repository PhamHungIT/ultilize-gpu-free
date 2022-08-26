import os
import logging

__logger__ = logging.getLogger()

def shutdown(time=None):
    __logger__.warning('Shutdown require sudo, if password input is needed, the program will hang!')
    command = 'sudo shutdown -h now'
    if time:
        command = f'sudo shutdown -h +{time}'
    os.system(command)
    return