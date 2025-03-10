import pandas as pd
import torch, sys, os, argparse
from pfhuman.autosome import AutosomeDataProcessor, AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomeTrainer
from pfhuman.bhi import BHIFirstLayersBlock, BHICoreBlock
from pfhuman.unlockdna import UnlockDNACoreBlock
from pfhuman.prixfixe import PrixFixeNet
#from pfhuman.evaluation import evaluate_predictions
from tqdm import tqdm

defaults = {'batch_size':256}

parser = argparse.ArgumentParser()
parser.add_argument("model_arch",choices=['cnn', 'rnn', 'attn'])
parser.add_argument("test_fold",type=str,choices=['1','2','3','4','5'])
parser.add_argument("num_epochs",type=int, default=80)
parser.add_argument("seed",type=int)
parser.add_argument("kwargs",nargs='*')
args = parser.parse_args()

kwargs = defaults.copy()
for kv in args.kwargs:
    key, value = kv.split('=')
    if key in kwargs:  # Only update known variables
        kwargs[key] = type(defaults[key])(value)  # Convert to correct type

if args.num_epochs > 200:
    print("#epochs and seed may have been mixed up")
    sys.exit()

TRAIN_DATA_PATH = f"/scratch/jqian1/gosai_data/train_no_{args.test_fold}.txt"
VALID_DATA_PATH = f"/scratch/jqian1/gosai_data/fold_{args.test_fold}.txt"
TRAIN_BATCH_SIZE = kwargs['batch_size']
N_PROCS = 4
VALID_BATCH_SIZE = 4096
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH))*2//TRAIN_BATCH_SIZE+1
SEQ_SIZE = 230
CUDA_DEVICE_ID = 0
lr = 0.001 if args.model_arch == 'attn' else 0.005 # 0.001 for attention layers in coreBlock
generator = torch.Generator()
generator.manual_seed(args.seed)

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
    seqsize=SEQ_SIZE,
    generator=generator
)

if args.model_arch == 'attn':
    first = AutosomeFirstLayersBlock(in_channels=5,
                                    out_channels=256,
                                    seqsize=SEQ_SIZE)
else:
    first = BHIFirstLayersBlock(
        in_channels = 5,
        out_channels = 320,
        seqsize = SEQ_SIZE,
        kernel_sizes = [9, 15],
        pool_size = 1,
        dropout = 0.2
    )
if args.model_arch == 'cnn':
    core = AutosomeCoreBlock(in_channels=first.out_channels,
                out_channels =64,
                seqsize=SEQ_SIZE)
elif args.model_arch == 'rnn':
    core = BHICoreBlock(
    in_channels = first.out_channels,
    out_channels = 320,
    seqsize = SEQ_SIZE,
    lstm_hidden_channels = 320,
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout1 = 0.2,
    dropout2 = 0.5
)
else:
    core = UnlockDNACoreBlock(
            in_channels = first.out_channels, out_channels= first.out_channels, seqsize = SEQ_SIZE, n_blocks = 4, kernel_size = 15, rate = 0.1, num_heads = 8)

final = AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                 seqsize=SEQ_SIZE)
model = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)

trainer = AutosomeTrainer(
    model,
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"),
    model_dir=f"models/test_{args.test_fold}/{args.model_arch}_{args.seed}",
    dataprocessor=dataprocessor,
    num_epochs=args.num_epochs,
    lr = lr)
trainer.fit()

'''
model.load_state_dict(torch.load(f'models/test_{args.test_fold}/{args.model_arch}_{args.seed}/model_best.pth', weights_only=True))
model.eval()
predictor = AutosomePredictor(model=model, model_pth=f"models/test_{args.test_fold}/{args.model_arch}_{args.seed}/model_best.pth", device=torch.device(f"cuda:0"))
test_df = pd.read_csv('/scratch/jqian1/test.txt', header=None, sep='\t')

pred_expr = []
for seq in tqdm(test_df.iloc[:, 0]):
    pred_expr.append(predictor.predict(seq))
evaluate_predictions(pred_expr, discard_public_leaderboard_indices=False)
'''