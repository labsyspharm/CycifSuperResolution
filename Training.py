import argparse
import os
import pickle
import sys

import lightning.pytorch.loggers
import lightning.pytorch.strategies
import lightning.pytorch.tuner
import torch.utils.data
import numpy.random
import torch
import lightning.pytorch as lp

from utils import CustomDataloader

numpy.random.seed(1)


class CycifSuperResolutionModel(lp.LightningModule):
    def __init__(self, args):
        super.__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1,3,2),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(3,1,2)
        )

    def training_step(self, batch):
        x, y = batch
        x = self.model(x)

        return x

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optim], [torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=5, verbose=True, threshold=0.05)]


class CycifDataModule(lp.LightningDataModule):
    def __init__(self, args):
        all_files = os.listdir(os.path.join(args.input, "input"))
        numpy.random.shuffle(all_files)
        train_f = int(args.frac_train * len(all_files))
        rest = (1 - args.frac_train)/2
        rest *= len(all_files)
        rest = int(rest)
        self.base_path = args.input
        self.files_train = all_files[0:train_f]
        self.files_test = all_files[train_f:train_f+rest]
        self.files_val = all_files[train_f+rest:]
        self.batch_size = args.batch_size
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = True

    def train_dataloader(self):
        return CustomDataloader(self.base_path, self.files_train)

    def test_dataloader(self):
        return CustomDataloader(self.base_path, self.files_test)

    def val_dataloader(self):
        return CustomDataloader(self.base_path, self.files_val)


def set_callbacks(args):
    output = list()

    if args.is_test_run:
        output.append(lp.callbacks.BatchSizeFinder(mode="binsearch"))
        output.append(lp.callbacks.LearningRateFinder(min_lr=0.0000001, max_lr=0.9))
    else:
        output.append(lp.callbacks.ModelCheckpoint(dirpath=args.output, monitor="val/loss", filename="epoch-{epoch:02d}_{val/loss:.4f}"))
        output.append(lp.callbacks.LearningRateMonitor(logging_interval="step"))

    return output


def pipeline(args):
    trainer = None

    if args.is_test_run:
        trainer = lp.trainer.Trainer(callbacks=set_callbacks(args),
                                 accelerator="gpu",
                                 devices=args.gpu_string,
                                 strategy=lp.strategies.SingleDeviceStrategy(device="cuda:" + args.gpu_string.split(",")[0])
                                     )
    else:
        trainer = lp.trainer.Trainer(logger=lp.loggers.TensorBoardLogger(args.tensorboard_path, name=args.title),
                                 callbacks=set_callbacks(args),
                                 accelerator="gpu",
                                 devices=args.gpu_string,
                                 strategy="auto")

    model = CycifSuperResolutionModel(args)

    data = CycifDataModule(args)

    trainer.fit(model, train_dataloaders=data)


def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("input", type=str, help="Base path of training data.")
    output.add_argument("--tensorboard-path", type=str, help="Path to store the tensorboard data.", default="./tensorboard")
    output.add_argument("--is-test-run", action="store_true", help="This run does not run training but hyper parameter setting before training.")
    output.add_argument("--gpu-string", type=str, help="Comma separated string with availabble gpus eg 0,1", default="1")
    output.add_argument("--frac-train", type=float, help="Fraction of dataset to use as training.", default=0.9)
    output.add_argument("--batch-size", type=int, help="Batch size for training.", default=32)

    output = output.parse_args(sys.argv[1:])

    if not os.path.exists(output.tensorboard_path):
        os.makedirs(output.tensorboard_path, exist_ok=True)

    return output


if __name__ == "__main__":
    args = parse_args()
    pipeline(args)
