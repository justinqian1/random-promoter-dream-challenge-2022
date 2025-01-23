import pandas as pd
import torch, sys, os, argparse
from prixfixe.autosome import AutosomeDataProcessor, AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomeTrainer, AutosomePredictor
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.prixfixe import PrixFixeNet
from prixfixe.evaluation import evaluate_predictions
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("model_arch",type=str)
parser.add_argument("num_epochs",type=int)
parser.add_argument("seed",type=int)

args = parser.parse_args()
if args.num_epochs > args.seed:
    print("#epochs and seed may have been mixed up")
    sys.exit()
if args.model_arch not in {"cnn", "rnn", "attn"}:
    print("model must be 'cnn', 'rnn', or 'attn'; model is instead {args.model_arch}")
    sys.exit()

TRAIN_DATA_PATH = "data/train_subset.txt" #change filename to actual training data
VALID_DATA_PATH = "data/val.txt" #change filename to actual validaiton data
TRAIN_BATCH_SIZE = 1024 # replace with 1024, if 1024 doesn't fit in gpu memory, decrease by order of 2 (512,256)
N_PROCS = 4
VALID_BATCH_SIZE = 4096
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH))//TRAIN_BATCH_SIZE
BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH))//VALID_BATCH_SIZE
PLASMID_PATH = "data/plasmid.json"
SEQ_SIZE = 150
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
    plasmid_path=PLASMID_PATH,
    seqsize=SEQ_SIZE,
    generator=generator
)

if args.model_arch == ('cnn' or 'rnn'):
    first = BHIFirstLayersBlock(
        in_channels = dataprocessor.data_channels(),
        out_channels = 320,
        seqsize = dataprocessor.data_seqsize(),
        kernel_sizes = [9, 15],
        pool_size = 1,
        dropout = 0.2
    )
else:
    first = AutosomeFirstLayersBlock(in_channels=dataprocessor.data_channels(),
                out_channels=256,
                seqsize=dataprocessor.data_seqsize())
if args.model_arch == 'cnn':
    core = AutosomeCoreBlock(in_channels=first.out_channels,
                out_channels =64,
                seqsize=first.infer_outseqsize())
elif args.model_arch == 'rnn':
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
else:
    core = UnlockDNACoreBlock(
            in_channels = first.out_channels, out_channels= first.out_channels, seqsize = dataprocessor.data_seqsize(), n_blocks = 4, kernel_size = 15, rate = 0.1, num_heads = 8)

final = AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                 seqsize=core.infer_outseqsize())
model = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)

trainer = AutosomeTrainer(
    model,
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"),
    model_dir=f"../../../../../../scratch/st-cdeboer-1/justin/models/{args.model_arch}_{args.seed}",
    dataprocessor=dataprocessor,
    num_epochs=args.num_epochs,
    lr = lr)
trainer.fit()

model.load_state_dict(torch.load(f'../../../../../../scratch/st-cdeboer-1/justin/models/{args.model_arch}_{args.seed}/model_best.pth', weights_only=True))
model.eval()
predictor = AutosomePredictor(model=model, model_pth=f"../../../../../../scratch/st-cdeboer-1/justin/models/{args.model_arch}_{args.seed}/model_best.pth", device=torch.device(f"cuda:0"))
test_df = pd.read_csv('data/test.txt', header=None, sep='\t')

pred_expr = []
for seq in tqdm(test_df.iloc[:, 0]):
    pred_expr.append(predictor.predict(seq))
evaluate_predictions(pred_expr, discard_public_leaderboard_indices=False)