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

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

class NormalAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams 

        self.encoder = nn.Sequential(
            # input (nc) x 256 x 256
            nn.Conv2d(hparams.nc, hparams.nfe, 4, 2, 1),
            nn.BatchNorm2d(hparams.nfe),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe) x 128 x 128
            nn.Conv2d(hparams.nfe, hparams.nfe*2, 4, 2, 1),
            nn.BatchNorm2d(hparams.nfe*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*2) x 64 x 64
            nn.Conv2d(hparams.nfe*2, hparams.nfe*4, 4, 2, 1),
            nn.BatchNorm2d(hparams.nfe*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*4) x 32 x 32
            nn.Conv2d(hparams.nfe*4, hparams.nfe*8, 4, 2, 1),
            nn.BatchNorm2d(hparams.nfe*8),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*8) x 16 x 16
            nn.Conv2d(hparams.nfe*8, hparams.nfe*16, 4, 2, 1),
            nn.BatchNorm2d(hparams.nfe*16),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*16) x 8 x 8
            nn.Conv2d(hparams.nfe*16, hparams.nfe*32, 4, 2, 1),
            nn.BatchNorm2d(hparams.nfe*32),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*32) x 4 x 4,
            nn.Conv2d(hparams.nfe*32, hparams.nz, 4, 2, 1),
            nn.BatchNorm2d(hparams.nz),
            nn.LeakyReLU(0.2, inplace=True),
            # output (nz) x 2 x 2
        )

        self.decoder = nn.Sequential(             
            # input (nz) x 2 x 2
            nn.ConvTranspose2d(hparams.nz, hparams.nfd * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nfd * 32),
            nn.ReLU(True),

            # input (nfd*32) x 4 x 4
            nn.ConvTranspose2d(hparams.nfd*32, hparams.nfd * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hparams.nfd * 16),
            nn.ReLU(True),

            # input (nfd*16) x 8 x 8
            nn.ConvTranspose2d(hparams.nfd * 16, hparams.nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 8),
            nn.ReLU(True),

            # input (nfd*8) x 16 x 16
            nn.ConvTranspose2d(hparams.nfd * 8, hparams.nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 4),
            nn.ReLU(True),

            # input (nfd*4) x 32 x 32
            nn.ConvTranspose2d(hparams.nfd * 4, hparams.nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd * 2),
            nn.ReLU(True),

            # input (nfd*2) x 64 x 64
            nn.ConvTranspose2d(hparams.nfd * 2, hparams.nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hparams.nfd),
            nn.ReLU(True),

            # input (nfd) x 128 x 128
            nn.ConvTranspose2d(hparams.nfd, hparams.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output (nc) x 256 x 256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            # draw weights from normal distribution
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            # initializing gamma and beta for BN
            nn.init.constant_(self.bias.data, 0)

    def prepare_data(self):

        transform = Transform(MEAN.tolist(), STD.tolist(), self.hparams)

        # retrieve normal cases of COVIDx dataset from COVID-Net paper
        self.train_ds = COVIDxNormal("train", self.hparams.data_root)
        self.test_ds = COVIDxNormal("test", self.hparams.data_root, transform=transform.test)

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
    
    model = NormalAE(hparams)
    model.apply(NormalAE.weights_init)

    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.img_size, hparams.img_size), device="cpu")

    trainer = Trainer(
        logger=logger, 
        gpus=hparams.gpus, 
        max_epochs=hparams.max_epochs
    )
    
    trainer.fit(model)
    trainer.test(model)

    timestamp = datetime.datetime.now().strftime(format="%d_%m_%Y_%H:%M:%S")
    if not os.path.exists(hparams.model_dir):
        os.makedirs(hparams.model_dir)
    
    save_pth = os.path.join(hparams.model_dir, "autoencoder_" + timestamp + ".pth")
    torch.save(model.state_dict(), save_pth)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data", help="Data root directory, where train and test folders are located")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory for saving trained models")
    parser.add_argument("--log_name", type=str, default="autoencoder", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--img_size", type=int, default=256, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=8, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images, e.g. 3 for RGB images")
    parser.add_argument("--nz", type=int, default=1024, help="Size of latent codes after encoders, i.e. number of feature maps in latent representation")
    parser.add_argument("--nfe", type=int, default=32, help="Number of feature maps in encoders")
    parser.add_argument("--nfd", type=int, default=32, help="Number of of feature maps in decoder")
    parser.add_argument("--aug_min_scale", type=float, default=0.75, help="Minimum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_max_scale", type=float, default=1.0, help="Maximum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_rot", type=float, default=5, help="Degrees arg for RandomRotation")
    parser.add_argument("--aug_bright", type=float, default=0.2, help="Brightness arg for ColorJitter")
    parser.add_argument("--aug_cont", type=float, default=0.1, help="Contrast arg for ColorJitter")

    args = parser.parse_args()
    main(args)
