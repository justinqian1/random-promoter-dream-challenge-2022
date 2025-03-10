import pandas as pd
import numpy as np
import torch, os, argparse, copy
from prixfixe.autosome import AutosomeDataProcessor, AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.prixfixe import PrixFixeNet

parser = argparse.ArgumentParser()
parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
parser.add_argument("seeds", type=int, nargs="+", help="List of initialization seeds")

args = parser.parse_args()

print("Received seeds:", args.seeds) 

TRAIN_DATA_PATH = "/scratch/jqian1/test_small.txt" #not used
VALID_DATA_PATH = "/scratch/jqian1/other_train.txt" #change filename to actual validaiton data
SEED = 1
TRAIN_BATCH_SIZE = 1024 # replace with 1024, if 1024 doesn't fit in gpu memory, decrease by order of 2 (512,256)
N_PROCS = 4
VALID_BATCH_SIZE = 4096
#BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH,header=None))//TRAIN_BATCH_SIZE
#BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH,header=None))//VALID_BATCH_SIZE
PLASMID_PATH = "/scratch/jqian1/plasmid.json"
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
    #batch_per_epoch=BATCH_PER_EPOCH,
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    plasmid_path=PLASMID_PATH,
    seqsize=SEQ_SIZE,
    generator=generator
)

if args.arch=='cnn':
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

elif args.arch=='rnn':
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

else: # attn
    first = AutosomeFirstLayersBlock(in_channels=6,
                                    out_channels=256,
                                    seqsize=150)

    core = UnlockDNACoreBlock(
        in_channels = first.out_channels, out_channels= first.out_channels, seqsize = 150, n_blocks = 4,
                                        kernel_size = 15, rate = 0.1, num_heads = 8)

    final = AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                    seqsize=core.infer_outseqsize())

models=[]
for seed in args.seeds:
    generator = torch.Generator().manual_seed(seed)
    model = PrixFixeNet(
            first=first,
            core=core,
            final=final,
            generator=generator
        )
    model.load_state_dict(torch.load(f'models_2m/{args.arch}_{seed}/model_best.pth', weights_only=True))
    models.append(copy.deepcopy(model))  

for model in models:
    model = model.to(device)
    model = model.eval()

def revcomp(seq: str):
    complement = str.maketrans("ACGT","TGCA")
    return seq.translate(complement)[::-1]

with torch.inference_mode():
    seq2var={}
    var2seq={}
    seq2expr={}
    for batch in dataprocessor.prepare_valid_dataloader():
        X = batch["x"]
        y = batch["y"]
        seq = batch["seq"]
        X = X.to(device)
        #y = y.float().to(device)
        model_preds=[]
        for model in models:
            model_preds.append(model.forward(X))
        combined = torch.stack(model_preds)
        combined=combined.cpu().numpy()
        var = np.var(combined,axis=0)
        for i in range(var.size):
            if X[i][4,1].item()==0.0:
                seq2expr.update({seq[i]:y[i].item()})
            else:
                seq[i] = revcomp(seq[i])
            if (seq[i] not in seq2var) or (seq2var[seq[i]] < var[i]):
                seq2var.update({seq[i]:var[i]})

    for seq in seq2var.keys():
        var2seq.update({seq2var[seq]:seq})

keys = list(var2seq.keys())
keys.sort(reverse=True)
print(keys[:10])

file = f'/scratch/jqian1/new_train/{args.arch}_{args.seeds}.txt'

with open(file,'w') as f:
    for i in range(100000):
        seq = var2seq.get(keys[i])
        #seq = seq[40:]
        f.write(seq+'\t'+str(seq2expr.get(seq))+'\n')
