import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from args import parse_args
from data import COVIDxNormal, random_split
from transforms import Transform
from unet import UNet
from utils import save_model

# normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


class NormalAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.unet = UNet(in_channels=hparams.nc, out_channels=hparams.nc)

    def forward(self, x):
        x = self.unet(x)
        return x

    def setup(self, mode):
        transform = Transform(MEAN.tolist(), STD.tolist(), self.hparams)

        # retrieve "normal" cases of COVIDx dataset from COVID-Net paper
        self.train_ds = COVIDxNormal(
            "train", self.hparams.data_root, self.hparams.dataset_dir, transform=transform.train
        )
        self.test_ds = COVIDxNormal(
            "test", self.hparams.data_root, self.hparams.dataset_dir, transform=transform.test
        )

        # define at which indices to plot during training
        num_train_batches = len(self.train_ds) // self.hparams.batch_size
        self.train_plot_indices = np.linspace(
            0, num_train_batches, self.hparams.num_plots_per_epoch, dtype=int
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))

    def plot(self, x, output, prefix, n=8):
        """Saves a plot of n images from input and output batch
        x         input batch
        output    output batch
        prefix    prefix of plot
        n         number of pictures to compare
        """

        if self.hparams.batch_size < n:
            raise IndexError("You are trying to plot more images than your batch contains!")

        # denormalize images
        denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        x = [denormalization(i) for i in x[:n]]
        output = [denormalization(i) for i in output[:n]]

        # make grids and save to logger
        grid_top = vutils.make_grid(x, nrow=n)
        grid_bottom = vutils.make_grid(output, nrow=n)
        grid = torch.cat((grid_top, grid_bottom), 1)

        name = f"{prefix}/input_reconstr_images"
        self.logger.experiment.add_image(name, grid)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        output = self(imgs)
        loss = F.mse_loss(output, imgs)

        # plot input and reconstructed images
        if batch_idx in self.train_plot_indices:
            self.plot(imgs, output, "train")

        logs = {f"train/loss": loss}
        return {f"loss": loss, "log": logs}

    def test_step(self, batch, batch_idx):
        imgs, _ = batch
        output = self(imgs)
        loss = F.mse_loss(output, imgs)

        # plot at beginning of epoch
        if batch_idx == 0:
            self.plot(imgs, output, "test")

        logs = {f"test/loss": loss}
        return {f"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x[f"test_loss"] for x in outputs]).mean()
        logs = {f"test/avg_loss": avg_loss}
        return {f"avg_test_loss": avg_loss, "log": logs}


def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=hparams.log_name)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # create model and print detailed summary with estimated network size
    model = NormalAE(hparams)
    summary(model, (hparams.nc, hparams.img_size, hparams.img_size), device="cpu")

    trainer = Trainer(
        logger=logger,
        gpus=hparams.gpus,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        max_epochs=hparams.max_epochs,
        weights_summary=None,
    )

    trainer.fit(model)
    trainer.test(model)
    save_model(model, hparams.models_dir, hparams.log_name)
    print("done. have a good day!")


if __name__ == "__main__":

    args = parse_args()
    main(args)
