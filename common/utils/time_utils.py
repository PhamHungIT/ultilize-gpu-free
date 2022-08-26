from datetime import datetime


def time_diff(first, second):
    seconds_in_day = 24 * 60 * 60
    difference = first - second
    minute, second = divmod(difference.days * seconds_in_day + difference.seconds, 60)
    return minute, second


def now():
    return datetime.utcnow()
