import pandas as pd
import numpy as np
import torch, sys, os
from prixfixe.autosome import AutosomeDataProcessor, AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomePredictor
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.prixfixe import PrixFixeNet
from prixfixe.evaluation import evaluate_predictions
from tqdm import tqdm

TRAIN_DATA_PATH = "data/test_small.txt" #change filename to actual training data
VALID_DATA_PATH = "data/test2.txt" #change filename to actual validaiton data
SEED = 314159
TRAIN_BATCH_SIZE = 1024 # replace with 1024, if 1024 doesn't fit in gpu memory, decrease by order of 2 (512,256)
N_PROCS = 4
VALID_BATCH_SIZE = 4096
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH,header=None))//TRAIN_BATCH_SIZE
BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH,header=None))//VALID_BATCH_SIZE
PLASMID_PATH = "data/plasmid.json"
SEQ_SIZE = 150
#NUM_EPOCHS = 2 #replace with 80
CUDA_DEVICE_ID = 0
device=torch.device(f"cuda:{CUDA_DEVICE_ID}")
#lr = 0.005 # 0.001 for attention layers in coreBlock
generator = torch.Generator()
generator.manual_seed(SEED) 

dataprocessor = AutosomeDataProcessor(
    path_to_training_data=TRAIN_DATA_PATH,
    path_to_validation_data=VALID_DATA_PATH,
    train_batch_size=TRAIN_BATCH_SIZE,
    batch_per_epoch=BATCH_PER_EPOCH,
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    plasmid_path=PLASMID_PATH,
    seqsize=SEQ_SIZE,
    generator=generator
)

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

first = BHIFirstLayersBlock(
    in_channels = 6,
    out_channels = 320,
    seqsize = 150,
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout = 0.2
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

test_df = pd.read_csv('data/test2.txt', header=None, sep='\t')

seq_to_expr={}
for i in range(test_df.shape[0]):
    seq_to_expr.update({test_df.iloc[i,0]:test_df.iloc[i,1]})

cnn = cnn.to(device)
cnn.load_state_dict(torch.load(f'/scratch/st-cdeboer-1/justin/models/cnn_{SEED}/model_best.pth', weights_only=True))
cnn = cnn.eval()
rnn = rnn.to(device)
rnn.load_state_dict(torch.load(f'/scratch/st-cdeboer-1/justin/models/rnn_{SEED}/model_best.pth', weights_only=True))
rnn = rnn.eval()
attn = attn.to(device)
attn.load_state_dict(torch.load(f'/scratch/st-cdeboer-1/justin/models/attn_{SEED}/model_best.pth', weights_only=True))
attn = attn.eval()

def revcomp(seq: str):
    complement = str.maketrans("ACGT","TGCA")
    return seq.translate(complement)[::-1]

with torch.inference_mode():
    seq2var={}
    var2seq={}
    for batch in dataprocessor.prepare_valid_dataloader():
        #remember: this thing does 2 passes, since it's sampling from 38 total
        #y_pred, y = self._evaluate(batch)
        X = batch["x"]
        #y = batch["y"]
        seq = batch["seq"]
        X = X.to(device)
        #y = y.float().to(device)
        cnn_pred = cnn.forward(X)
        rnn_pred = rnn.forward(X)
        attn_pred = attn.forward(X)
        combined=torch.stack((cnn_pred,rnn_pred,attn_pred),dim=0)
        combined=combined.cpu().numpy()
        var = np.var(combined,axis=0)
        for i in range(var.size):
            if seq[i][1]=='A':
                seq[i] = revcomp(seq[i])
            if (seq[i] not in seq2var) or (seq2var[seq[i]] < var[i]):
                seq2var.update({seq[i]:var[i]})
                var2seq.update({var[i]:seq[i]})

keys = list(var2seq.keys())
keys.sort(reverse=True)
print(keys[:100])

file = f'/scratch/st-cdeboer-1/justin/new_train_{SEED}.txt'

with open(file,'w') as f:
    for i in range(100):
        seq = var2seq.get(keys[i])
        seq = seq[40:]
        f.write(seq+'\t'+seq_to_expr.get(seq).astype(str)+'\n')
