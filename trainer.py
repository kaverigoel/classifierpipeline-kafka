import torch
import argparse
import pytorch_lightning as pl
# from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import importlib
import wandb


model_name = "ResNet"

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--data_dir',
                    help='Absolute path to root of training & testing directory',
                    type=str)
parser.add_argument('--out_channels',
                    help='Number of output classes',
                    type=int)
parser.add_argument('--num_workers',
                    help='Number of data workers for the DataLoader',
                    default=4,
                    type=int)
parser.add_argument('--loss',
                    help='Loss function used by the model',
                    default='cross_entropy',
                    type=str)
parser.add_argument('--max_nb_epochs',
                    help='Max number of epochs the models will be run for',
                    default=100,
                    type=int)
parser.add_argument('--val_split',
                    help='Split between training and validation data',
                    default=0.8,
                    type=float)
parser.add_argument('--batch_size',
                    help='Batch size for the input',
                    default=32,
                    type=int)
parser.add_argument('--aug',
                    help='If True will augment training data',
                    default=True,
                    type=bool)


def main(args):
    torch.manual_seed(0)
    # Initialize wandb and other model callbacks
    # time = datetime.now().strftime("%H:%M:%S")
    run_name = f"loss_fn_{args.loss}_100Epochs_TrainRun"
    wandb.init(name=run_name)
    model_checkpoint = ModelCheckpoint(dirpath=f"logs/{run_name}", monitor='val_loss',
                                       save_top_k=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                                   strict=True, verbose=True, mode="min")
    # Define model and trainer
    model_obj = getattr(importlib.import_module(
        'models'), model_name)
    model = model_obj(model_name, params=args)

    # Define the pl trainer
    trainer = pl.Trainer(gpus=1,
                         max_epochs=args.max_nb_epochs,
                         callbacks=[model_checkpoint, early_stopping])
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # print(args.batch_size)
    model = main(args)
