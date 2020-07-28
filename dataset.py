from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import config
import torch

class QuoraDataset(Dataset):

    def __init__(self,train=True):
        self.train = train
        if self.train==True:
            self.data = pd.read_csv(config.TRAIN_DATA_PATH).iloc[:20000,1].values
            self.labels = pd.read_csv(config.TRAIN_DATA_PATH).iloc[:20000,2].values
        else:
            ind = 260000
            self.data = pd.read_csv(config.TEST_DATA_PATH).iloc[ind:ind+10000,1].values
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.n_samples = len(self.data)

    def _truncate_tokens(self,data):
        data = list(data)
        l = self.max_len //2
        return data[:l] + data[-l:]

    def _get_tokens(self,text):
        return self.tokenizer.encode(text,add_special_tokens=True)
    
    def _padding(self,data):
        data=list(data)
        return np.array(data + [0] * (self.max_len - len(data)))
    
    def _get_attention_mask(self,data):
        return np.where(data!=0,1,0)

    def __getitem__(self,index):
        tokens = self._get_tokens(self.data[index])
        if len(tokens) > self.max_len:
            tokens = self._truncate_tokens(tokens)
        input_ids = self._padding(tokens)
        attention_mask = self._get_attention_mask(input_ids)
        if self.train==True:
            label = self.labels[index]
            res = {'input_ids':input_ids,'attention_mask':attention_mask,'label':label}
        else:
            res = {'input_ids':input_ids,'attention_mask':attention_mask}
        return res

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    data = QuoraDataset()
    print(data[0])