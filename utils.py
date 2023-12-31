import torch
import os
import pickle
import torch.utils.tensorboard
import torch.utils.data
import tqdm
import numpy


class CustomDataloader(torch.utils.data.Dataset):
    def __init__(self, base_path, files, is_uint16, device="cuda:0"):
        self.basepath = base_path
        self.files = files
        self.device = device
        self.uint16 = is_uint16

    def __getitem__(self, item):
        x, y = None, None

        with open(os.path.join(self.basepath, "input", self.files[item]), "rb") as f:
            x = pickle.load(f)

        with open(os.path.join(self.basepath, "output", self.files[item]), "rb") as f:
            y = pickle.load(f)

        if self.uint16:
            x = x.astype(numpy.int32) / 2**16
            y = x.astype(numpy.int32) / 2**16
        else:
            x = x / 255
            y = y / 255

        x = torch.FloatTensor(x).to(self.device).reshape((1, 512, 512))
        y = torch.FloatTensor(y).to(self.device).reshape((1, 512, 512))

        return x, y

    def __len__(self):
        return len(self.files)


def training_loop(model, optim, dataloader, loss_fn, epoch_n=0, **kwargs):
    l = 0.0
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader), desc="batch", leave=False):
        x, y = data
        optim.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()

        l += loss.item()

        if "tensorboard" in kwargs:
            kwargs["tensorboard"].add_scalar("epoch_{}/train".format(epoch_n), loss.item(), batch_idx)
    l /= (len(dataloader) * kwargs["batch_size"])

    return l


def testing_loop(model, dataloader, loss_fn, epoch_n=0, **kwargs):
    l = 0.0
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader), desc="batch", leave=False):
        x, y = data

        pred = model(x)

        loss = loss_fn(pred, y)
        l += loss.item()
        if "tensorboard" in kwargs:
            kwargs["tensorboard"].add_scalar("epoch_{}/test".format(epoch_n), loss.item(), batch_idx)

    l /= (len(dataloader) * kwargs["batch_size"])

    return l


def add_example_images(args, model, epoch, tb: torch.utils.tensorboard.SummaryWriter):
    for image in os.listdir(args.example_images):
        with open(os.path.join(args.example_images, image), "rb") as f:
            i = pickle.load(f)

            if args.is_uint16:
                i = torch.FloatTensor(i.astype(numpy.array_api.int32))
            else:
                i = torch.FloatTensor(i)

            i = torch.FloatTensor(i).to(args.device)
            tb.add_image(image, model(i) * 255, epoch, dataformats="HW")
