import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as rnn
import math
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from config import UNIGRAM_DIM, BIGRAM_DIM, WINDOW_SIZE, UNI_NUM, FEATURE_DIM

# NLC
class LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, weight):
        super(LSTM, self).__init__()
        self.hidden_dim = out_dim
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size = in_dim, hidden_size = out_dim, 
                            batch_first = True, num_layers = self.num_layers, dropout = 0.1)
        self.linear = nn.Linear(out_dim, UNI_NUM)
        self.linear.weight = weight
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        self.final_dropout = torch.nn.Dropout(p=0.1)
        
    def forward(self, x, c, h):
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x, (h, c))
        output = self.final_dropout(output)
        return self.logsoftmax(self.linear(output))

class NPMT(nn.Module):
    def __init__(self, uni_embedding=None, vocab=None, MAX_SEGMENT = 4, hrate = 0.1):
        super(NPMT, self).__init__()
        self.vocab = vocab
        self.uni_embedding = nn.Embedding(np.shape(uni_embedding)[0], np.shape(uni_embedding)[1])        
        self.uni_embedding.weight = nn.Parameter(torch.Tensor(uni_embedding))

        self.LSTM = LSTM(UNIGRAM_DIM, FEATURE_DIM, self.uni_embedding.weight)

        self.loginf = 1000000.0
        self.dropout = torch.nn.Dropout(p=0.1)
        self.eos = 4
        
        self.LSTMCell = torch.nn.LSTMCell(UNIGRAM_DIM, FEATURE_DIM)
        
        self.y_start = Parameter(torch.Tensor(1, FEATURE_DIM).cuda())        
        self.y_input0 = torch.nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.hlinear = torch.nn.Linear(FEATURE_DIM, FEATURE_DIM * self.LSTM.num_layers)

        self.MAX_SEGMENT = MAX_SEGMENT
        self.hrate = hrate
        
    def forward(self, x, lens, sy = None):
        if sy is None:
            sy = []
        lens = lens.data.tolist()
        batch_size = x.size()[0]
        maxT = max(lens)
        #NL
        x = x[:,WINDOW_SIZE:-WINDOW_SIZE]
        y = x
        x = self.uni_embedding(x)
        y_start = self.y_start.repeat(batch_size, 1)

        cell_h_start = Variable(x.data.new(batch_size, FEATURE_DIM).zero_()).contiguous() 
        cell_c_start = Variable(x.data.new(batch_size, FEATURE_DIM).zero_()).contiguous() 
        
        c_start = Variable(x.data.new(self.LSTM.num_layers, batch_size, FEATURE_DIM).zero_()).contiguous() 

        schedule = []
        for j_start in range(maxT):
            j_len = min(self.MAX_SEGMENT, maxT-j_start)
            j_end = j_start + j_len
            schedule.append((j_start, j_len, j_end))        
                
        #log_probability
        logpy = [[Variable(torch.DoubleTensor([-self.loginf]*batch_size).cuda())
                  for _ in range(self.MAX_SEGMENT+1)] for _ in range(maxT)]
        
        for j_start, j_len, j_end in schedule:      
            if j_start == 0:
                h, c = cell_h_start, cell_c_start
                h, c = self.LSTMCell(self.dropout(y_start), (h, c))
            else:  
                h, c = self.LSTMCell(self.dropout(x[:, j_start-1, :]), (h, c))
            
            #h = cell_h_start
            
            drop_h = self.hrate * h
            
            y_input0 = self.y_input0(drop_h).view(batch_size, 1, FEATURE_DIM)
            
            y_input = torch.cat((y_input0, x[:, j_start:j_end, :]), dim = 1).contiguous()
            
            y_input = self.dropout(y_input)
            
            y_output = y[:, j_start:j_end].contiguous().view(-1, int(j_len), 1)
            
            h_start = self.hlinear(drop_h).view(batch_size, self.LSTM.num_layers, FEATURE_DIM).transpose(0, 1).contiguous()
            h_start = torch.nn.functional.tanh(h_start)
            
            t_prob = self.LSTM(y_input, c_start, h_start).double()
            
            t_vec_whole = t_prob[:, :-1, :].gather(dim = 2, index = y_output).view(-1, int(j_len))
            
            t_vec = Variable(torch.zeros(batch_size).double().cuda())
            
            for j in range(j_start, j_end):
                t_vec = t_vec + t_vec_whole[:, j-j_start].contiguous().view(-1)
                logpy[j_start][j-j_start+1] = t_vec + t_prob[:, j-j_start+1, self.eos]
            
        #log_probability
        alpha = [Variable(torch.DoubleTensor([-self.loginf]*batch_size).cuda())
                  for _ in range(maxT+1)]
        alpha[0] = Variable(torch.DoubleTensor([0.0]*batch_size).cuda())
        
        for j in range(maxT+1):
            for j_start in range(max(1, j - self.MAX_SEGMENT + 1), j+1):                    
                logprob = alpha[j_start-1] + logpy[j_start-1][j-j_start+1]
                alpha[j] =  self.log_sum_exp(alpha[j], logprob)
        
        ret = 0.0
        
        if self.training:
            count = 0
            for i in range(batch_size):
                ret = ret - alpha[lens[i]][i]
                count += lens[i]
            ret = ret/count
            
            ret2 = 0.0
            count = 0
            for i in range(len(sy)):
                for a, b in sy[i]:
                    ret2 = ret2 - logpy[a][b][i]
                    count += b
            if count > 0:
                ret2 = ret2/count
                    
            return ret, ret2
            
        else:
            for i in range(batch_size):
                ret = ret - alpha[lens[i]][i]
            out_str = []
            for i in range(batch_size):
                out_str.append(self.print_best_path(lens[i], y[i,:], i, logpy))
                
            return ret, out_str
    
    def log_sum_exp(self, a, b):
        ret = nn.functional.log_softmax(torch.stack((a, b)), dim=0)[0,:]
        return a - ret
    
    def print_best_path(self, ylen, yref, i, logpy):
        alpha = [-self.loginf]*(ylen+1)
        prev = [-1]*(ylen+1)
        alpha[0] = 0.0
        for j in range(ylen+1):
            for j_start in range(max(1, j - self.MAX_SEGMENT + 1), j+1):
                logprob = alpha[j_start-1] + float(logpy[j_start-1][j-j_start+1][i])
                if logprob > alpha[j]:
                    alpha[j] = logprob
                    prev[j] = j_start-1
        j = ylen
        out_str = "|"
        while j > 0:
            prev_j = prev[j]
            for k in range(j, prev_j, -1):
                out_str = self.vocab[int(yref[k-1])] + out_str
            out_str = "|" + out_str
            j = prev_j

        return out_str 
        
if __name__ == "__main__":
    test()