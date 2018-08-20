import pandas as pd
import numpy as np
from config import WINDOW_SIZE

class BucketedDataIterator():
    def __init__(self, df, df2, BATCH_SIZE, BATCH_SIZE2, num_buckets=8):               
        df['syi'] = ''
        df['syl'] = ''
        self.df = df
        self.df2= df2
        self.total = len(df)
        self.total2 = len(df2)
        df_sort = df.sort_values('length').reset_index(drop=True)
        df_sort2 = df2.sort_values('length').reset_index(drop=True)
        self.size = self.total // num_buckets
        self.size2 = self.total2 // num_buckets
        self.dfs = []
        self.dfs2 = []
        for bucket in range(num_buckets - 1):
            self.dfs.append(df_sort.ix[bucket*self.size: (bucket + 1)*self.size - 1])
            self.dfs2.append(df_sort2.ix[bucket*self.size2: (bucket + 1)*self.size2 - 1])
        self.dfs.append(df_sort.ix[(num_buckets-1)*self.size:])
        self.dfs2.append(df_sort2.ix[(num_buckets-1)*self.size2:])
        
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.cursor2 = np.array([0] * num_buckets)
        self.pos = 0
        self.shuffle()
        self.shuffle2()
        self.epochs = 0
        self.BATCH_SIZE = BATCH_SIZE
        self.BATCH_SIZE2 = BATCH_SIZE2

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
    
    def shuffle2(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs2[i] = self.dfs2[i].sample(frac=1).reset_index(drop=True)
            self.cursor2[i] = 0
    
    def next_batch(self):
        if np.any(self.cursor + self.BATCH_SIZE + 1 > self.size):
            self.epochs += 1
            self.shuffle()
        
        if np.any(self.cursor2 + self.BATCH_SIZE2 + 1 > self.size2):
            self.shuffle2()
        
        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + self.BATCH_SIZE - 1]
        
        res2 = self.dfs2[i].ix[self.cursor2[i]:self.cursor2[i] + self.BATCH_SIZE2 - 1]
        
        res = res.append(res2, ignore_index = True)
        
        res = res.sort_values('length', ascending = False)
        
        biwords = list(map(lambda x: list(map(int, x.split(","))), res['biwords'].tolist()))
        words = list(map(lambda x: list(map(int, x.split(","))), res['words'].tolist()))
        
        syi = list(map(lambda x: list(map(int, x.replace(',',' ').split())), res['syi'].tolist()))
        syl = list(map(lambda x: list(map(int, x.replace(',',' ').split())), res['syl'].tolist()))
        
        sy = [[(a,b) for (a, b) in zip(i, l)] for (i, l) in zip(syi, syl)]
        
        self.cursor[i] += self.BATCH_SIZE
        self.cursor2[i] += self.BATCH_SIZE2
        
        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        bi_x = np.zeros([self.BATCH_SIZE + self.BATCH_SIZE2, maxlen + WINDOW_SIZE*2 + 1], dtype=np.int32)
        for i, x_i in enumerate(bi_x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2 + 1] = biwords[i]
        x = np.zeros([self.BATCH_SIZE + self.BATCH_SIZE2, maxlen + WINDOW_SIZE*2], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i] + WINDOW_SIZE*2] = words[i]

        return bi_x, x, res['length'].values, sy