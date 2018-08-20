# -*- coding: utf-8 -*-

import os

DATA = 'pku'

BIGRAM = False
MAX_LEN = 30
CLIP = 0.1
UNI_NUM = 8676
FEATURE_DIM = 256
WINDOW_SIZE = 1
NUM_CARD = 0
BATCH_SIZE = 256
BATCH_SIZE2 = 16 #8 for msr, cityu, 2 for as, 16 for pku
LAMBDA = 0.2 #msr 0.03, cityu, as 0.07
hrate = 0.05
LR = 0.005
MAX_SEGMENT = 4
BATCH_SIZE = 512

UNIGRAM_DIM = 256
BIGRAM_DIM = 50

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIR, 'models')
DATA_DIR = os.path.join(DIR, 'data')
RESULT_DIR = os.path.join(DIR, 'results')
GIGA_PATH = os.path.join(DATA_DIR, '_small_giga.txt')
TRAIN_PATH = os.path.join(DATA_DIR, '_' + DATA + '.txt')
UNIGRAM_PATH = os.path.join(MODEL_DIR, 'unigram' + str(UNIGRAM_DIM) + '.txt')
BIGRAM_PATH = os.path.join(MODEL_DIR, 'bigram' + str(BIGRAM_DIM) + '.txt')
SUPER_PATH = os.path.join(DATA_DIR, 'supervised_' + DATA + '.txt')
TEST_GOLD2 = os.path.join(DATA_DIR, DATA + '_training.txt')
TEST_FILE2 = os.path.join(DATA_DIR, DATA + '.txt')
TEST_GOLD = os.path.join(DATA_DIR, DATA + '_test_gold.txt')
TEST_FILE = os.path.join(DATA_DIR, DATA + '_test.txt')

RESULT_FILE = os.path.join(RESULT_DIR, 'result'+str(NUM_CARD))
IMPROVED_RESULT_FILE = os.path.join(RESULT_DIR, 'improved_result'+str(NUM_CARD))

LOG_PATH = os.path.join(DIR, 'log')