# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import pandas as pd
from config import MAX_LEN, WINDOW_SIZE

class Data_index(object):
    def __init__(self, Vocabs):
        self.VOCABS = Vocabs
    
    def to_bi_index(self, words):
        uni_idx = []
        for _ in range(WINDOW_SIZE):
            words.append('</s>')
            words.insert(0, '</s>')
        for word in words:
            if word in self.VOCABS.uni2idx:
                uni_idx.append(self.VOCABS.uni2idx[word])
            else:
                uni_idx.append(self.VOCABS.uni2idx['<OOV>'])
        
        left = words[:]
        left.insert(0, '</s>')
        right = words[:]
        right.append('</s>')

        bi_idx = []
        
        for current_word, next_word in zip(left, right):
            word = current_word + next_word
            if word in self.VOCABS.bi2idx:
                bi_idx.append(self.VOCABS.bi2idx[word])
            else:
                bi_idx.append(self.VOCABS.bi2idx['<OOV>'])
        
        return bi_idx, uni_idx
    
    def rindex(self, mylist):
        if u'<PUNC>' not in mylist or mylist[::-1].index(u'<PUNC>') == len(mylist) - 1:
            return MAX_LEN
        else:
            return len(mylist) - mylist[::-1].index(u'<PUNC>')
    
    def process_lines(self, path):
        lines = []
        with open(path) as train_file:
            for line in train_file:          
                if line.strip() == '':
                    continue

                words = []
                ret = []
                for line_t in line.strip().split(' '):
                    words.append(line_t)

                while len(words) > MAX_LEN + 5:
                    length = self.rindex(words[:MAX_LEN])
                    bi_idx, uni_idx = self.to_bi_index(words[:length])
                    bi_idx = ','.join(map(str, bi_idx))
                    uni_idx = ','.join(map(str, uni_idx))
                    lines.append([bi_idx, uni_idx, length])          
                    words = words[length:]
                else:
                    length = len(words)
                    bi_idx, uni_idx = self.to_bi_index(words[:length])
                    bi_idx = ','.join(map(str, bi_idx))
                    uni_idx = ','.join(map(str, uni_idx))
                    lines.append([bi_idx, uni_idx, length])
        
        return pd.DataFrame(lines, columns=['biwords', 'words', 'length'])

    def process_supervised_lines(self, path):
        lines = []
        with open(path) as train_file:
            for line in train_file:          
                if line.replace('| ', '').strip() == '':
                    continue

                words = []
                ret = []
                for line_t in line.strip().split(' '):
                    words.append(line_t)
                
                sy = []
                prev_i = 0
                i = 0
                while i < len(words):
                    if words[i] == '|':
                        sy.append((prev_i, i - prev_i))
                        prev_i = i
                        del words[i]
                    else:
                        i += 1
                
                start = 0
                while len(words) > MAX_LEN + 5:
                    length = self.rindex(words[:MAX_LEN])
                    bi_idx, uni_idx = self.to_bi_index(words[:length])
                    bi_idx = ','.join(map(str, bi_idx))
                    uni_idx = ','.join(map(str, uni_idx))
                    syi = [i - start for (i, l) in sy if start + l <= i + l <= length]
                    syi = ','.join(map(str, syi))
                    syl = [l for (i, l) in sy if start + l <= i + l <= length]
                    syl = ','.join(map(str, syl))
                    start = length
                    lines.append([bi_idx, uni_idx, length, syi, syl])          
                    words = words[length:]
                else:
                    length = len(words)
                    bi_idx, uni_idx = self.to_bi_index(words[:length])
                    bi_idx = ','.join(map(str, bi_idx))
                    uni_idx = ','.join(map(str, uni_idx))
                    syi = [i - start for (i, l) in sy if start + l <= i + l <= length]
                    syi = ','.join(map(str, syi))
                    syl = [l for (i, l) in sy if start + l <= i + l <= length]
                    syl = ','.join(map(str, syl))
                    lines.append([bi_idx, uni_idx, length, syi, syl])
        
        return pd.DataFrame(lines, columns=['biwords', 'words', 'length', 'syi', 'syl'])