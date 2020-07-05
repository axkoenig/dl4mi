__author__ = 'Alexander Koenig, Li Nguyen'

import datetime
import os
from argparse import ArgumentParser

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

from data import COVIDxNormal, random_split
from transforms import Transform
from args import parse_args
from unet import UNet

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

    def prepare_data(self):

        transform = Transform(MEAN.tolist(), STD.tolist(), self.hparams)

        # retrieve normal cases of COVIDx dataset from COVID-Net paper
        self.train_ds = COVIDxNormal("train", self.hparams.data_root, self.hparams.dataset_dir)
        self.test_ds = COVIDxNormal("test", self.hparams.data_root, self.hparams.dataset_dir, transform=transform.test)

        # further split train into train and val
        train_split = 0.95
        train_size = int(train_split * len(self.train_ds))
        val_size = len(self.train_ds) - train_size
        self.train_ds, self.val_ds = random_split(self.train_ds, [train_size, val_size])

        # apply correct transforms
        self.train_ds.transform = transform.train
        self.val_ds.transform = transform.test

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
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
        
        name = f"{prefix}_input_reconstr_images"
        self.logger.experiment.add_image(name, grid)

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
        imgs, _ = batch
        output = self(imgs)
        loss = F.mse_loss(output, imgs)

        # plot input, mixed and reconstructed images at beginning of epoch
        if plot and batch_idx == 0:
            self.plot(imgs, output, prefix)

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
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=hparams.log_name)
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    # create model and print detailed summary with estimated network size
    model = NormalAE(hparams)
    summary(model, (hparams.nc, hparams.img_size, hparams.img_size), device="cpu")

    trainer = Trainer(
        logger=logger, 
        gpus=hparams.gpus, 
        max_epochs=hparams.max_epochs,
        weights_summary=None,
    )
    
    trainer.fit(model)
    trainer.test(model)

    timestamp = datetime.datetime.now().strftime(format="%d_%m_%Y_%H:%M:%S")
    if not os.path.exists(hparams.model_dir):
        os.makedirs(hparams.model_dir)
    
    save_pth = os.path.join(hparams.model_dir, "autoencoder_" + timestamp + ".pth")
    torch.save(model.state_dict(), save_pth)

if __name__ == "__main__":

    args = parse_args()
    main(args)
