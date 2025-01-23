import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('data/train.txt', header=None,sep='\t')
count = 0
print(f"dimensions: {df.size}\n")
for num in tqdm(df.iloc[:,1]):
    if np.abs(num - np.round(num)) < 0.0001:
        count+=1
print(count)
