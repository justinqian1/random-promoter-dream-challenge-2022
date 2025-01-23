import pandas as pd
import torch
import os
from prixfixe.autosome import AutosomeDataProcessor, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomeTrainer, AutosomePredictor
from prixfixe.bhi import BHIFirstLayersBlock
from prixfixe.prixfixe import PrixFixeNet
from prixfixe.evaluation import evaluate_predictions
from tqdm import tqdm

TRAIN_DATA_PATH = "data/train.txt" #change filename to actual training data
VALID_DATA_PATH = "data/val.txt" #change filename to actual validaiton data
SEED = 314159
TRAIN_BATCH_SIZE = 1024 # replace with 1024, if 1024 doesn't fit in gpu memory, decrease by order of 2 (512,256)
N_PROCS = 4
VALID_BATCH_SIZE = 4096
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH))//TRAIN_BATCH_SIZE
BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH))//TRAIN_BATCH_SIZE
PLASMID_PATH = "data/plasmid.json"
SEQ_SIZE = 150
NUM_EPOCHS = 2 #replace with 80
CUDA_DEVICE_ID = 0
lr = 0.005 # 0.001 for attention layers in coreBlock
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
    in_channels = dataprocessor.data_channels(),
    out_channels = 320,
    seqsize = dataprocessor.data_seqsize(),
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout = 0.2
    )

core = AutosomeCoreBlock(in_channels=first.out_channels,
                         out_channels =64,
                         seqsize=first.infer_outseqsize())

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
    model_dir=f"../../../../../../scratch/st-cdeboer-1/justin/models/cnn_{SEED}",
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCHS,
    lr = lr)
trainer.fit()

predictor = AutosomePredictor(model=model, model_pth=f"../../../../../../scratch/st-cdeboer-1/justin/models/cnn_{SEED}/model_best.pth", device=torch.device(f"cuda:0"))
test_df = pd.read_csv('data/test.txt', header=None, sep='\t')

pred_expr = []
for seq in tqdm(test_df.iloc[:, 0]):
    pred_expr.append(predictor.predict(seq))
evaluate_predictions(pred_expr, discard_public_leaderboard_indices=False)
