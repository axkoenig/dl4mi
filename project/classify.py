from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchsummary import summary
from torchvision import utils
from torchvision.datasets import ImageFolder

from autoencoder import NormalAE
from dataset import COVIDx, random_split
from transforms import Transform


def plot_dataset(dataset, n=6):
    # retrieve random images from dataset
    choice = np.random.randint(len(dataset), size=n)
    subset = Subset(dataset, choice)
    images = [x[0] for x in subset]
    labels = [x[1] for x in subset]

    # denormalize for visualization
    denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
    images = [denormalization(i) for i in images]

    # make grid and plot
    grid = utils.make_grid(images)
    label_string = []
    for label in labels:
        if "0":
            label_string.append("normal")
        elif "1":
            label_string.append("pneumonia")
        elif "2":
            label_string.append("COVID-19")

    title = "Labels are: " + str(label_string)
    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.show()


# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


class Classifier(pl.LightningModule):
    def __init__(self, hparams, autoencoder):
        super().__init__()
        self.hparams = hparams
        self.autoencoder = autoencoder

        # number of neurons in last dense layers
        self.nnd = self.hparams.nfc * 4 * (self.hparams.img_size // 4 ** 3) ** 2

        self.classifier = nn.Sequential(
            # input (nc) x 256 x 256
            nn.Conv2d(self.hparams.nc, self.hparams.nfc, 4, 2, 1),
            nn.BatchNorm2d(self.hparams.nfc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # input (nfc) x 64 x 64
            nn.Conv2d(self.hparams.nfc, self.hparams.nfc * 2, 4, 2, 1),
            nn.BatchNorm2d(self.hparams.nfc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # input (nfc*2) x 16 x 16
            nn.Conv2d(self.hparams.nfc * 2, self.hparams.nfc * 4, 4, 2, 1),
            nn.BatchNorm2d(self.hparams.nfc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # input (nfc*4) x 4 x 4
            nn.Flatten(),
            nn.Linear(self.nnd, self.nnd, bias=True),
            nn.Linear(self.nnd, self.nnd, bias=True),
            nn.Linear(self.nnd, 3, bias=True),
            nn.Softmax(),
        )

    def forward(self, x):
        # create anomaly map
        reconstructed = self.autoencoder(x)
        anomaly = x - reconstructed

        # classify anomaly map
        prediction = self.classifier(anomaly)

        return {"reconstructed": reconstructed, "anomaly": anomaly, "prediction": prediction}

    def prepare_data(self):

        transform = Transform(MEAN.tolist(), STD.tolist(), self.hparams)

        # retrieve COVIDx dataset from COVID-Net paper
        self.train_ds = COVIDx("train")
        self.test_ds = COVIDx("test", transform=transform.test)

        # configure sampler to rebalance training set
        weights = 1 / torch.Tensor(
            [
                self.train_ds.counter["normal"],
                self.train_ds.counter["pneumonia"],
                self.train_ds.counter["COVID-19"],
            ]
        )
        self.sampler = WeightedRandomSampler(weights, self.hparams.batch_size)

        # further split train into train and val
        train_split = 0.95
        train_size = int(train_split * len(self.train_ds))
        val_size = len(self.train_ds) - train_size
        self.train_ds, self.val_ds = random_split(self.train_ds, [train_size, val_size])

        # apply correct transforms
        self.train_ds.transform = transform.train
        self.val_ds.transform = transform.test

        if self.hparams.debug:
            plot_dataset(self.train_ds)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            sampler=self.sampler,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

    def plot(self, x, r, a, prefix, n=4):
        """Plots n triplets of (original image, reconstr. image, anomaly map)

        Args:
            x (tensor): Batch of input images
            r (tensor): Batch of reconstructed images
            a (tensor): Batch of anomaly maps
            prefix (str): Prefix for plot name 
            n (int, optional): How many triplets to plot. Defaults to 16.

        Raises:
            IndexError: If n exceeds batch size
        """

        if x.shape[0] < n:
            raise IndexError("You are attempting to plot more images than your batch contains!")

        # denormalize images
        denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        x = [denormalization(i) for i in x[:n]]
        r = [denormalization(i) for i in r[:n]]
        # a = [denormalization(i) for i in a[:n]]

        # create empty plot and send to device
        plot = torch.tensor([], device=x[0].device)

        for i in range(n):

            grid = utils.make_grid([x[i], r[i], a[i]], 1)
            plot = torch.cat((plot, grid), 2)

            # add offset between image triplets
            if n > 1 and i < n - 1:
                border_width = 6
                border = torch.zeros(plot.shape[0], plot.shape[1], border_width, device=x[0].device)
                plot = torch.cat((plot, border), 2)

        name = f"{prefix}_input_reconstr_anomaly_images"
        self.logger.experiment.add_image(name, plot)

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="val", plot=True)

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix="test", plot=True)

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "test")

    def _shared_eval(self, batch, batch_idx, prefix="", plot=False):

        imgs, labels = batch
        out = self(imgs)
        loss = F.cross_entropy(out["prediction"], labels)

        # plot input, mixed and reconstructed images at beginning of epoch
        if plot and batch_idx == 0:
            self.plot(imgs, out["reconstructed"], out["anomaly"], prefix)

        # add underscore to prefix
        if prefix:
            prefix = prefix + "_"

        logs = {f"{prefix}loss": loss}
        return {f"{prefix}loss": loss, "log": logs}

    def _shared_eval_epoch_end(self, outputs, prefix):
        avg_loss = torch.stack([x[f"{prefix}_loss"] for x in outputs]).mean()
        logs = {f"avg_{prefix}_loss": avg_loss}
        return {f"avg_{prefix}_loss": avg_loss, "log": logs}


def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name="classifier")

    # load pretrained autoencoder
    autoencoder = NormalAE(hparams)
    autoencoder.load_state_dict(torch.load(hparams.ae_pth))
    autoencoder.eval()

    # create classifier
    model = Classifier(hparams, autoencoder)

    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.img_size, hparams.img_size), device="cpu")

    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--img_size", type=int, default=256, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=8, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nfc", type=int, default=8, help="Number of feature maps in classifier")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    parser.add_argument("--ae_pth", type=str, default="models/autoencoder_27_05_2020_15:17:25.pth", help="Path of trained autoencoder")
    parser.add_argument("--nz", type=int, default=1024, help="Autoencoder param - Size of latent code")
    parser.add_argument("--nfe", type=int, default=32, help="Autoencoder param - Number of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Autoencoder param - Number of feature maps in decoder")
    parser.add_argument("--aug_min_scale", type=float, default=0.8, help="Minimum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_max_scale", type=float, default=1.0, help="Maximum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_rot", type=float, default=5, help="Degrees arg for RandomRotation")
    parser.add_argument("--aug_bright", type=float, default=0.1, help="Brightness arg for ColorJitter")
    parser.add_argument("--aug_cont", type=float, default=0.1, help="Contrast arg for ColorJitter")
    
    args = parser.parse_args()
    main(args)
