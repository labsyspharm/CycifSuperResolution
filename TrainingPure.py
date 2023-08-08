import argparse
import os
import sys

import torch
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import utils


class CycifSRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = torch.nn.ReLU()(y)
        y = self.conv2(x)
        y = torch.nn.ReLU()(y)

        return y

def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("input", type=str, help="Path to root directory of dataset.")
    output.add_argument("output", type=str, help="Path to store models and checkpoints.")
    output.add_argument("--batch-size", type=int, help="Batch size for training.", default=32)
    output.add_argument("--lr-annealing", choices=["cyclic", "plateau", "exp"], help="Use lr annelaing while training.")
    output.add_argument("--device", type=str, help="Device to use in training.", default="cuda:0")
    output.add_argument("--tensorboard-dir", type=str, help="Path to store tensorboard data. None if do not use tensorboard", default=None)
    output.add_argument("--epochs", type=int, default=100)

    output = output.parse_args(sys.argv[1:])

    if not os.path.exists(os.path.join(output.input, "input")):
        print("Input path does not contain \"input\" folder.")
        sys.exit(1)

    if not os.path.exists(os.path.join(output.input, "output")):
        print("Input path does not contain \"output\" folder.")
        sys.exit(1)

    if not os.path.exists(os.path.join(output.output)):
        os.makedirs(output.output)

    if output.tensorboard_dir is not None and not os.path.exists(os.path.join(output.tensorboard_dir)):
        os.makedirs(output.tensorboard_dir)

    return output


def main():
    args = parse_args()
    print(args)

    model = CycifSRModel()
    model = model.to(args.device)

    filenames = os.listdir(os.path.join(args.input, "input"))

    dataloader_train = torch.utils.data.DataLoader(
        utils.CustomDataloader(args.input, filenames[0:int(0.9*len(filenames))], args.device),
                                             batch_size=args.batch_size, shuffle=True, drop_last=True,
                                             )
    dataloader_test = torch.utils.data.DataLoader(
        utils.CustomDataloader(args.input, filenames[int(0.9*len(filenames)):], args.device),
                                             batch_size=args.batch_size, shuffle=True, drop_last=True,
                                             )

    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    lr = None
    tb = torch.utils.tensorboard.SummaryWriter(args.tensorboard_dir) if args.tensorboard_dir is not None else None

    loss = torch.nn.MSELoss()

    if args.lr_annealing == "exp":
        lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, verbose=True, gamma=0.01)
    elif args.lr_annealing == "cyclic":
        lr = torch.optim.lr_scheduler.CyclicLR(optim, mode="exp_range", base_lr=0.00001, max_lr=0.01)
    elif args.lr_annealing == "plateau":
        lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, patience=4)

    kwargs = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_annealing": lr,
        "tensorboard": tb
    }

    for epoch in tqdm.tqdm(range(args.epochs), desc="Epoch: "):
        loss_train = utils.training_loop(model, optim, dataloader_train, loss, epoch, **kwargs)

        with torch.no_grad():
            loss_test = utils.testing_loop(model, dataloader_test, loss, epoch, **kwargs)

        if tb is not None:
            tb.add_scalar("train/loss_sum", loss_train, epoch)
            tb.add_scalar("test/loss_sum", loss_test, epoch)

        if lr is not None:
            lr.step(loss_test)


if __name__ == "__main__":
    main()
