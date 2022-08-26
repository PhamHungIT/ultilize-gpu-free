# Need to declare langdetect==1.0.8 in requirement.txt
from langdetect import detect
from langdetect import detect_langs

'''
langdetect supports 55 languages
af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,
hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,
pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw
'''


def language_detect(input_text):
    try:
        lang = detect(input_text)
        if lang == 'en':
            return 'en'
        return 'vi'
    except Exception as e:
        print(f"language detect error: {e}")
        return 'vi'


def languages_detect(input_text):
    # List of language (lang:prob) with highest prob
    return detect_langs(input_text)
