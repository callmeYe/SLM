import torch
import numpy as np
from torch.autograd import Variable
from chinese import chinese
from config import RESULT_FILE, IMPROVED_RESULT_FILE, TEST_GOLD, TEST_FILE, MAX_LEN, BIGRAM

class Eval(object):
    def __init__(self, da_idx, BATCH_SIZE):   
        self.da_idx = da_idx
        self.BATCH_SIZE = BATCH_SIZE
        self.init()
        
    def init(self):
        self.improved_P = 0.6
        self.improved_R = 0.6
        self.improved_F = 0.6

        self.P = 0.6
        self.R = 0.6
        self.F = 0.6
        
        self.test_loss = 100
        
    def proc(self, ans_sentence):
        for ch in '的了和与就在很也都将你我他她它要这上':
            ans_sentence = ans_sentence.replace(ch, '  '+ch+'  ')
        while '    ' in ans_sentence:
            ans_sentence = ans_sentence.replace('    ', '  ')
        word_lst = ['了解', '为了', '除了',
                    '与其', '与否', '参与',
                    '成就', '就要',
                    '现在', '正在', '存在', '所在', '在于',
                    '很多', '很难', '很快',
                    '即将', '必将', '将来',
                    '你们', '他们', '其他', '其它', '它们','她们', '我们', '我国', '自我',
                    '主要', '需要', '要求', '重要', '只要', '还要',
                    '这里', '这次', '这样', '这种', '这是', '这些', '这个',
                    '上午', '上年', '上海', '上市', '以上']
        for ch in word_lst:
            ans_sentence = ans_sentence.replace(ch[0] + '  ' + ch[1:], '  '+ch+'  ')
            ans_sentence = ans_sentence.replace(ch[:-1] + '  ' + ch[-1], '  '+ch+'  ')
            ans_sentence = ans_sentence.replace('  '.join(ch), '  '+ch+'  ')
        for moon in ['１', '２', '３', '４', '５', '６', '７', '８', '９', '１０', '１１', '１２',
                    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']:
            ans_sentence = ans_sentence.replace('年  '+moon+'月', '年'+moon+'月')
            ans_sentence = ans_sentence.replace('月  '+moon, '月'+moon)
            ans_sentence = ans_sentence.replace(moon + '  月', moon+'月')

        while '    ' in ans_sentence:
            ans_sentence = ans_sentence.replace('    ', '  ')
        while '  \n' in ans_sentence:
            ans_sentence = ans_sentence.replace('  \n', '\n')
        return ans_sentence

    def deal(self, ans):
        ret = ""
        for i in range(len(ans)):
            if str(type(ans[i])) == "<class 'list'>":
                x = ans[i][0]
                num_lst = ans[i][1]
                count = ans[i][2]
                line = ans[i][3]
                xxcount = 0

                for ele in x:
                    if ele != '':
                        j = 0
                        while j < len(ele):
                            if ele[j:j+5] in ['<NUM>', '<ENG>']:
                                ret += num_lst[xxcount]
                                count += len(num_lst[xxcount])
                                xxcount += 1
                                j += 5
                            elif ele[j:j+5] == '<OOV>':
                                ret += line[count]
                                count += 1
                                j += 5
                            else:
                                ret += ele[j]
                                count += 1
                                j += 1
                        ret += '  '
            else:
                ret += ans[i]
        return ret

    def transform(self, ans, NPMT):
        to_do_lst = []
        loss = 0
        loss_len = 0
        for i in range(len(ans)):
            ele = ans[i]
            if str(type(ele)) == "<class 'list'>":
                if len(ele[0]) <= 2:
                    ele[0] = [''.join(ele[0])]
                else:
                    to_do_lst.append(i)
            if len(to_do_lst) == self.BATCH_SIZE or (i == len(ans)-1 and len(to_do_lst) > 0):
                to_do_lst.sort(key=lambda x: len(ans[x][0]), reverse=True)
                len_lst = [len(ans[_][0]) for _ in to_do_lst]
                maxlen = max(len_lst)
                x_batch = []
                bi_x_batch = []
                length_batch = []   

                for j in to_do_lst:
                    words = ans[j][0][:]
                    length = len(words)
                    bi_idx, uni_idx = self.da_idx.to_bi_index(words)
                    uni_idx = uni_idx + [0]*(maxlen - len(ans[j][0]))
                    bi_idx = bi_idx + [0]*(maxlen - len(ans[j][0]))
                    x_batch.append(uni_idx)
                    bi_x_batch.append(bi_idx)
                    length_batch.append(len(ans[j][0]))

                for i in range(len(to_do_lst)):
                    loss_len += length_batch[i]

                x_batch = np.asarray(x_batch, dtype=np.int32)
                bi_x_batch = np.asarray(bi_x_batch, dtype=np.int32)
                length_batch = np.asarray(length_batch, dtype=np.int32)

                x = Variable(torch.LongTensor(x_batch).cuda(), volatile = True).view(len(to_do_lst), -1)
                if BIGRAM:
                    bi_x = VOCABS.bi_vectors[bi_x_batch]
                    bi_x = Variable(torch.Tensor(bi_x).cuda(), volatile = True).view(len(to_do_lst), -1,BIGRAM_DIM)
                else:
                    bi_x = None
                length = Variable(torch.LongTensor(length_batch).cuda(), volatile = True).view(-1)

                ret = NPMT(x, length)

                loss += float(ret[0])

                for j, outstr in zip(to_do_lst, ret[1]):
                    ans[j][0] = outstr.split('|')

                to_do_lst = []
        self.test_loss = loss / loss_len     
    
    def test_process(self, NPMT):
        NPMT.eval()
        ans = []
        for line in open(TEST_FILE, 'r').readlines():
            line = line.strip()
            sentence = []
            num_lst = []
            curr_num = ""
            count = 0
            start = 0
            for i in range(len(line)):
                _ = line[i]
                if len(sentence) >= MAX_LEN:             
                    if curr_num != "":
                        continue
                    ans.append([sentence[:], num_lst[:], start, line])
                    start = i
                    num_lst = []
                    sentence = []                
                if chinese(_) not in ['<PUNC>', '<NUM>', '<ENG>']:
                    if curr_num != "":
                        num_lst.append(curr_num)
                        curr_num = ""
                    sentence.append(_)
                elif chinese(_) in ['<NUM>', '<ENG>']:
                    if curr_num == '':
                        sentence.append(chinese(_))
                    curr_num += _
                elif chinese(_) == '<PUNC>':
                    if curr_num != "":
                        num_lst.append(curr_num)
                        curr_num = ''
                    if len(sentence) > 0:
                        ans.append([sentence[:], num_lst[:], start, line])
                    ans.append(_+'  ')
                    num_lst = []
                    sentence = []
                    start = i+1
            if len(sentence) > 0:
                if curr_num != "":
                    num_lst.append(curr_num)
                    curr_num = ''
                ans.append([sentence[:], num_lst[:], start, line])
                ans.append('  ')
            ans.append('\n')

        self.transform(ans, NPMT)
        
        fout = open(RESULT_FILE + '.txt', 'w')
        fout.write(self.deal(ans))
        fout.close()
    
    def improved_eval_process(self):
        improved = open(IMPROVED_RESULT_FILE + '.txt', 'w')
        my_ans = open(RESULT_FILE + '.txt', 'r').readlines()[:-1]
        gold_ans = open(TEST_GOLD, 'r').readlines()[:len(my_ans)]
        pred_seg = 0
        ans_seg = 0
        common_seg = 0
        for ans_sentence, pred_sentence in zip(my_ans, gold_ans):
            ans_sentence = self.proc(ans_sentence)
            improved.write(ans_sentence)
            ans = []
            for word in ans_sentence.strip().split('  '):                
                for i in range(len(word) - 1):
                    ans.append(-1)
                if word != '':
                    ans.append(word)

            pred = []
            for word in pred_sentence.strip().split('  '):            
                for i in range(len(word) - 1):
                    pred.append(-1)
                pred.append(word)

            for ans_word, pred_word in zip(ans, pred):
                if pred_word != -1:
                    pred_seg+=1
                if ans_word != -1:
                    ans_seg+=1
                if ans_word != -1 and pred_word == ans_word:
                    common_seg+=1
        print(ans_seg, pred_seg, common_seg)
        self.improved_P = common_seg/ans_seg
        self.improved_R = common_seg/pred_seg
        self.improved_F = 2/(1/self.improved_P +1/self.improved_R)     
        improved.close()
        print(self.improved_P, self.improved_R, self.improved_F)

    def eval_process(self):
        my_ans = open(RESULT_FILE + '.txt', 'r').readlines()[:-1]
        gold_ans = open(TEST_GOLD, 'r').readlines()[:len(my_ans)]
        pred_seg = 0
        ans_seg = 0
        common_seg = 0
        for ans_sentence, pred_sentence in zip(my_ans, gold_ans):
            ans = []
            for word in ans_sentence.strip().split('  '):                
                for i in range(len(word) - 1):
                    ans.append(-1)
                if word != '':
                    ans.append(word)

            pred = []
            for word in pred_sentence.strip().split('  '):            
                for i in range(len(word) - 1):
                    pred.append(-1)
                pred.append(word)

            for ans_word, pred_word in zip(ans, pred):
                if pred_word != -1:
                    pred_seg+=1
                if ans_word != -1:
                    ans_seg+=1
                if ans_word != -1 and pred_word == ans_word:
                    common_seg+=1
        print(ans_seg, pred_seg, common_seg)
        self.P = common_seg/ans_seg
        self.R = common_seg/pred_seg
        self.F = 2/(1/self.P+1/self.R)        
        print(self.P, self.R,self.F)