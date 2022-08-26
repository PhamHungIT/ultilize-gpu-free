import re


def validate_phone(phone):
    regex = "^((\+)?(84|0)(9|3|7|8|5|16|12|19)([0-9]{8}))$"
    r_phone = re.match(regex, phone)
    if r_phone:
        return True
    return False


def validate_mail(mail):
    regex = "^[a-z0-9]+([\._]?[a-z0-9][\._]?)+[@]\w+[.]\w{2,3}$"
    r_mail = re.match(regex, mail)
    if r_mail:
        return True
    return False
