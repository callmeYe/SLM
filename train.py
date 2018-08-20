import os
import datetime
import pandas as pd
import numpy as np
import torch
import model
import data_helpers
from torch.autograd import Variable
from voc import Vocab
from prepare_data_index import Data_index
from collections import defaultdict
from config import CLIP, BIGRAM_DIM, MAX_LEN, BIGRAM, DATA, TRAIN_PATH, SUPER_PATH, GIGA_PATH, NUM_CARD, BATCH_SIZE2, hrate, LAMBDA, LR, BATCH_SIZE, MAX_SEGMENT
from logger import Logger
from evaluate import Eval
import torch.optim as optim

print('init')

VOCABS = Vocab()
uni_embedding = VOCABS.uni_vectors
bi_embedding = VOCABS.bi_vectors
uni_num = VOCABS.uni_num
da_idx = Data_index(VOCABS)

train_data_iterator = data_helpers.BucketedDataIterator(
    da_idx.process_lines(TRAIN_PATH), da_idx.process_supervised_lines(SUPER_PATH), BATCH_SIZE - BATCH_SIZE2, BATCH_SIZE2)

evaluator = Eval(da_idx, 256)

torch.cuda.set_device(NUM_CARD)
NPMT = model.NPMT(uni_embedding = uni_embedding, vocab = VOCABS.idx2uni, MAX_SEGMENT=MAX_SEGMENT, hrate=hrate).cuda()

summary_count = 0
summary_path = './logs/' + DATA + '_' + str(MAX_SEGMENT)  + '_' + str(hrate) + '_'+ str(summary_count) 
while os.path.exists(summary_path):
    summary_count += 1
    summary_path = './logs/' + DATA + '_' + str(MAX_SEGMENT) + '_' + str(hrate) + '_'+ str(summary_count)
logger = Logger(summary_path)

name =  DATA + '_' + str(MAX_SEGMENT) + '_' + str(hrate) + '_'+ str(summary_count)

lr = LR

step = 0

best_F = 0.5

optimizer = optim.Adam(NPMT.parameters(), lr = LR)

total_loss1 = 0
total_loss2 = 0

while True: 
    if step % 400 == 0:
        loss = 0
        evaluator.test_process(NPMT)
        evaluator.eval_process()      
        evaluator.improved_eval_process()
        if best_F < evaluator.F:
            best_F = evaluator.F
            torch.save(NPMT, './save/model' + name)
    NPMT.train()
    bi_idx_batch, x_batch, seq_len_batch, sy = train_data_iterator.next_batch()
    x_batch = Variable(torch.LongTensor(x_batch).cuda())
    seq_len_batch = Variable(torch.LongTensor(seq_len_batch).cuda())

    NPMT.zero_grad()
    loss1, loss2 = NPMT(x_batch, seq_len_batch, sy)
    loss = loss1 + LAMBDA * loss2
    loss.backward()
    torch.nn.utils.clip_grad_norm(NPMT.parameters(), CLIP)
    
    if step > 800:
        optimizer.step()
    else:
        for p in NPMT.parameters():
            if p.grad is not None:
                p.data.add_(-16.0, p.grad.data)

    step += 1

    total_loss1 += float(loss1)
    total_loss2 += float(loss2)

    if step % 20 == 0:               
        info = {'loss/train': total_loss1/20, 'loss/train2': total_loss2/20, 'loss/test': float(evaluator.test_loss),
                'score/Pi1': evaluator.improved_P, 'score/Ri1': evaluator.improved_R, 'score/Fi1': evaluator.improved_F, 
                'score/P1': evaluator.P, 'score/R1': evaluator.R, 'score/F1': evaluator.F}
        
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step) 
        total_loss1 = 0   
        total_loss2 = 0
    
    if step % 3000 == 0:
        lr /= 10
        optimizer = optim.Adam(NPMT.parameters(), lr = lr)
    
    if step > 4200:
        print(name)
        print(best_F)
        del loss
        del NPMT
        break