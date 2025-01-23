import pandas as pd
import numpy as np
import torch, sys, os
from prixfixe.autosome import AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomePredictor
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.prixfixe import PrixFixeNet
from prixfixe.evaluation import evaluate_predictions
from tqdm import tqdm

SEED = 314159
CUDA_DEVICE_ID=0
generator = torch.Generator()
generator.manual_seed(SEED)

first = BHIFirstLayersBlock(
    in_channels = 6,
    out_channels = 320,
    seqsize = 150,
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout = 0.2
    )

core = AutosomeCoreBlock(in_channels=first.out_channels,
                         out_channels =64,
                         seqsize=first.infer_outseqsize())

final = AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                 seqsize=core.infer_outseqsize())

cnn = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
    )


core = BHICoreBlock(
    in_channels = first.out_channels,
    out_channels = 320,
    seqsize = first.infer_outseqsize(),
    lstm_hidden_channels = 320,
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout1 = 0.2,
    dropout2 = 0.5
    )
final = AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                 seqsize=core.infer_outseqsize())
rnn = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)


first = AutosomeFirstLayersBlock(in_channels=6,
                                   out_channels=256,
                                   seqsize=150)

core = UnlockDNACoreBlock(
    in_channels = first.out_channels, out_channels= first.out_channels, seqsize = 150, n_blocks = 4,
                                     kernel_size = 15, rate = 0.1, num_heads = 8)

final = AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                 seqsize=core.infer_outseqsize())
attn = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)

cnn_predictor = AutosomePredictor(model=cnn, model_pth=f'../../../../../../scratch/st-cdeboer-1/justin/models/cnn_{SEED}/model_best.pth', device=torch.device(f"cuda:0"))
rnn_predictor = AutosomePredictor(model=rnn, model_pth=f'../../../../../../scratch/st-cdeboer-1/justin/models/rnn_{SEED}/model_best.pth', device=torch.device(f"cuda:0"))
attn_predictor = AutosomePredictor(model=attn, model_pth=f'../../../../../../scratch/st-cdeboer-1/justin/models/attn_{SEED}/model_best.pth', device=torch.device(f"cuda:0"))
test_df = pd.read_csv('data/other_train.txt', header=None, sep='\t')

seq_to_expr={}
for i in range(test_df.shape[0]):
    seq_to_expr.update({test_df.iloc[i,0]:test_df.iloc[i,1]})

uncertainty = {}
for seq in test_df.iloc[:,0]:
    uncertainty.update({np.var([cnn_predictor.predict(seq),rnn_predictor.predict(seq),attn_predictor.predict(seq)]):seq})

keys = list(uncertainty.keys())
keys.sort(reverse=True)

file = f'../../../../../../scratch/st-cdeboer-1/justin/new_train_{SEED}.txt'

with open(file,'w') as f:
    for i in range(100000):
        seq = uncertainty.get(keys[i])
        f.write(seq+'\t'+seq_to_expr.get(seq).astype(str)+'\n')
