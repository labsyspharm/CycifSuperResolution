import argparse
import os
import pickle
import sys

import torch
import torch.utils.data
import torch.utils.tensorboard

import tqdm

import utils


class CycifSRModel(torch.nn.Module):
    def __init__(self):
        super(CycifSRModel, self).__init__()

        self.conv1 = self._create_autoencoder()
        self.conv2 = self._create_autoencoder()

    def forward(self, x):
        y = self.conv1(x)

        #y = self.conv2(y)

        return y

    def _create_down_block(self, i, o):
        s1 = torch.nn.Conv2d(i, o, 3, padding=1)
        s2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        return torch.nn.Sequential(s1, torch.nn.ReLU(), s2)

    def _create_up_block(self, i, o, k=3):
        s1 = torch.nn.ConvTranspose2d(in_channels=i, kernel_size=k, out_channels=o, stride=2)

        return torch.nn.Sequential(s1, torch.nn.ReLU())

    def _create_encoder(self):
        s1 = self._create_down_block(1, 128)
        s2 = self._create_down_block(128, 64)
        s3 = self._create_down_block(64, 32)

        return torch.nn.Sequential(s1, s2, s3)

    def _create_decoder(self):
        s1 = self._create_up_block(32, 32, k=2)
        s2 = self._create_up_block(32, 64, k=2)
        s3 = self._create_up_block(64, 128, k=2)

        return torch.nn.Sequential(s1, s2, s3)

    def _create_autoencoder(self):
        s1 = self._create_encoder()
        s2 = self._create_decoder()
        s3 = torch.nn.Conv2d(128, 1, 3, padding=1)

        return torch.nn.Sequential(s1, s2, s3)

def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("input", type=str, help="Path to root directory of dataset.")
    output.add_argument("output", type=str, help="Path to store models and checkpoints.")
    output.add_argument("--batch-size", type=int, help="Batch size for training.", default=32)
    output.add_argument("--lr-annealing", choices=["cyclic", "plateau", "exp"], help="Use lr annelaing while training.")
    output.add_argument("--device", type=str, help="Device to use in training.", default="cuda:0")
    output.add_argument("--tensorboard-dir", type=str, help="Path to store tensorboard data. None if do not use tensorboard", default=None)
    output.add_argument("--epochs", type=int, default=100)
    output.add_argument("--example-images", type=str, help="Example images to put in tensorboard log. None if no examples.", default=None)

    output = output.parse_args(sys.argv[1:])

    if output.tensorboard_dir is None and output.example_images is not None:
        print("Invalid parameter combination, if example-images is used you must give a tensorboard dir.")
        sys.exit(1)

    if not os.path.exists(os.path.join(output.input, "input")):
        print("Input path does not contain \"input\" folder.")
        sys.exit(1)

    if not os.path.exists(os.path.join(output.input, "output")):
        print("Input path does not contain \"output\" folder.")
        sys.exit(1)

    if not os.path.exists(os.path.join(output.output)):
        os.makedirs(output.output)

    if output.tensorboard_dir is not None and not os.path.exists(os.path.join(output.tensorboard_dir)):
        os.makedirs(output.tensorboard_dir, exist_ok=True)

    if not os.path.exists(os.path.join(output.output, "models")):
        os.makedirs(os.path.join(output.output, "models"), exist_ok=True)

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

    optim = torch.optim.Adam(model.parameters(), lr=0.01,)
    lr = None
    tb = torch.utils.tensorboard.SummaryWriter(args.tensorboard_dir) if args.tensorboard_dir is not None else None

    loss = torch.nn.MSELoss()

    if args.lr_annealing == "exp":
        lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, verbose=True, gamma=0.01)
    elif args.lr_annealing == "cyclic":
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        lr = torch.optim.lr_scheduler.CyclicLR(optim, mode="exp_range", base_lr=0.00001, max_lr=0.01)
    elif args.lr_annealing == "plateau":
        lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, patience=4)

    kwargs = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_annealing": lr,
        "tensorboard": tb
    }

    if args.example_images is not None:  # if we have example data we add it as step -1?
        for image in os.listdir(args.example_images):
            with open(os.path.join(args.example_images, image), "rb") as f:
                i = pickle.load(f)
                i = torch.FloatTensor(i)
                tb.add_image(image, i, -1)

    first = True
    for epoch in tqdm.tqdm(range(args.epochs), desc="Epoch: "):
        loss_train = utils.training_loop(model, optim, dataloader_train, loss, epoch, **kwargs)

        with torch.no_grad():
            loss_test = utils.testing_loop(model, dataloader_test, loss, epoch, **kwargs)

        if tb is not None:
            tb.add_scalar("sum/train_loss", loss_train, epoch)
            tb.add_scalar("sum/test_loss", loss_test, epoch)

        if lr is not None:
            lr.step(loss_test)

        if first and tb is not None:
            tb.add_graph(model, torch.FloatTensor(range(512*512)).reshape((1, 512, 512)).to(args.device))
            first = False

        torch.save(model, os.path.join(args.output, "models", "epoch_{}_loss_{}.pkl".format(epoch, loss_test)))

        if args.example_images is not None:
            utils.add_example_images(args, model, epoch, tb)


if __name__ == "__main__":
    main()
