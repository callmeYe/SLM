#!/usr/bin/env python
# -*- coding: utf-8 -*-
def chinese(uchar):
    '''判断一个unicode是否是汉字'''
    if u'\u4e00' <= uchar <= u'\u9fff':
        return uchar
    
    '''绝对顿开的标点符号有就是<PUNC>'''
    
    '''
    ０１２３４５６７８９0123456789％．＋＞ⅢⅣⅠⅡⅤ∶‰+㈨℃.
    '''
    if u'\uff10' <= uchar <= u'\uff19' or u'\u0030' <= uchar <= u'\u0039'\
    or u'\u2160' <= uchar <= u'\u2179' or uchar in u'％．＋＞∶‰+㈨℃.':
        return u'<NUM>'
    
    
    '''
    ｃｂａｇｆｅｄｋｉｈｏｎｍｌ
    ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＰＱＲＳＴＵＶＷＯＺ
    A-Z
    a-z
    α
    ＆
    '''
    if u'\uff41' <= uchar <= u'\uff5a' or u'\uff21' <= uchar <= u'\uff3a'\
    or u'\u0041' <= uchar <= u'\u005A' or u'\u0061' <= uchar <= u'\u007A'\
    or uchar in u'＆α':
        return u'<ENG>' 
    
    return u'<PUNC>'