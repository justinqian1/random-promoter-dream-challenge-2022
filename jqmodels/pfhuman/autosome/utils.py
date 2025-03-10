import json
from collections import Counter

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Generator
from pathlib import Path 

CODES: dict[str, int] = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

def n2id(n: str) -> int:
    return CODES[n.upper()]

def pad_sequence(seq, seqsize:int):
    total_pad = seqsize - len(seq)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return 'N' * left_pad + seq + 'N' * right_pad

class DataloaderWrapper:
    def __init__(self,
                 dataloader: DataLoader,
                 batch_per_epoch: int):
        self.batch_per_epoch = batch_per_epoch
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __len__(self):
        return self.batch_per_epoch
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)

    def __iter__(self):
        for _ in range(self.batch_per_epoch):
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)

class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str) -> torch.Tensor:
        seq_i = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq_i))
        code = F.one_hot(code, num_classes=5) # 5th class is N        
        code = code[:, :4].float() 
        return code.transpose(0, 1)

COMPL = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'N': 'N'
}

def n2compl(n: str) -> str:
    return COMPL[n.upper()]

def revcomp(seq: str):
    return "".join((n2compl(x) for x in reversed(seq)))

def add_rev(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    df['rev'] = 0
    revdf['rev'] = 1
    df = pd.concat([df, revdf]).reset_index(drop=True)
    return df

def preprocess_df(path: str | Path,  
                  seqsize: int):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns=['seq','expr']
    df['seq'] = df['seq'].apply(pad_sequence,args=(seqsize,))
    df = add_rev(df)
    return df

def initialize_weights(m: nn.Module, generator: Generator):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n), generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001, generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)