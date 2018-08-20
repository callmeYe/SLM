#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from config import UNIGRAM_DIM, BIGRAM_DIM, UNIGRAM_PATH, BIGRAM_PATH, BIGRAM

class Vocab(object):
    def __init__(self):      
        self.uni2idx = defaultdict(int)
        self.bi2idx = defaultdict(int)
        
        self.idx2uni = defaultdict(str)
        
        self.uni_vectors = None
        self.bi_vectors = None

        self.uni_num = 0
        
        self.load_data()

    def load_data(self):
        with open(UNIGRAM_PATH, 'r') as f:
            line = f.readline().strip().split(" ")
            N, dim = list(map(int, line))

            self.uni_vectors = []
            
            idx = 0
            self.uni2idx['<OOV>'] = idx
            self.idx2uni[idx] = '<OOV>'
            
            vector = np.zeros(UNIGRAM_DIM, dtype=np.float32)
            self.uni_vectors.append(vector)
            idx += 1
            
            
            for k in range(N):
                line = f.readline().strip().split(" ")
                self.uni2idx[line[0]] = idx
                self.idx2uni[idx] = line[0]
                vector = np.asarray(list(map(float, line[1:])), dtype=np.float32)
                self.uni_vectors.append(vector)
                idx += 1
            
            self.uni_num = idx
            self.idx2uni[idx] = '<EOS>'
            
            self.uni_vectors = np.asarray(self.uni_vectors, dtype=np.float32)
            
            print("uni_num", self.uni_num)
        
        if BIGRAM:
            with open(BIGRAM_PATH, 'r') as f:
                line = f.readline().strip().split(" ")
                N, dim = list(map(int, line))

                self.bi_vectors = []
                idx = 0
                self.bi2idx['<OOV>'] = idx
                vector = np.zeros(BIGRAM_DIM, dtype=np.float32)
                self.bi_vectors.append(vector)
                idx += 1

                for k in range(N):
                    line = f.readline().strip().split(" ")
                    self.bi2idx[line[0]] = idx
                    vector = np.asarray(list(map(float, line[1:])), dtype=np.float32)
                    self.bi_vectors.append(vector)
                    idx += 1

                self.bi_vectors = np.asarray(self.bi_vectors, dtype=np.float32)

                print("bi_num", idx)