import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchsummary import summary
from torchvision import utils
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.model_selection import KFold

from data import COVIDx, TransformableSubset
from transforms import Transform
from utils import calc_metrics, freeze, get_class_weights, save_model
from args import parse_args
from unet import MulittaskUNet

# normalization constants
MEAN = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
STD = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

# variables for rebalancing loss function
weight_train = None
weight_val = None


class Classifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.multitask_unet = MulittaskUNet(in_channels=hparams.nc, out_channels=hparams.nc)

        # variables to save model predictions
        self.gt_train = []
        self.pr_train = []
        self.gt_val = []
        self.pr_val = []

    def forward(self, x):
        return self.multitask_unet(x)

    def test_dataloader(self):
        transform = Transform(MEAN.tolist(), STD.tolist(), self.hparams)
        covidx_test = COVIDx(
            "test", self.hparams.data_root, self.hparams.dataset_dir, transform=transform.test,
        )

        return DataLoader(
            covidx_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),)

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
        imgs, labels = batch
        predictions, reconstruction = self(imgs)
        class_loss = F.cross_entropy(predictions, labels, weight=weight_train.cuda())
        rec_loss = F.mse_loss(reconstruction, imgs)
        loss = class_loss + self.hparams.alpha * rec_loss

        # reset predictions from last epoch
        if batch_idx == 0:
            self.gt_train = []
            self.pr_train = []

        # save labels and predictions for evaluation
        max_indices = torch.max(predictions, 1).indices
        self.gt_train += labels.tolist()
        self.pr_train += max_indices.tolist()

        logs = {f"train/loss": loss}
        return {f"loss": loss, "log": logs}

    def training_epoch_end(self, outputs):

        print(f"\n---> metrics for entire train epoch are: \n")
        metrics = calc_metrics(self.gt_train, self.pr_train, verbose=True)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {f"train/avg_loss": avg_loss}

        # tensorboard only saves scalars
        loggable_metrics = ["accuracy", "recall", "precision"]
        metrics = {f"train/{key}": metrics[key] for key in loggable_metrics}
        logs.update(metrics)

        return {"train/avg_loss": avg_loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val", plot=True)

    def validation_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test", plot=True)

    def test_epoch_end(self, outputs):
        return self._shared_eval_epoch_end(outputs, "test")

    def _shared_eval(self, batch, batch_idx, prefix, plot=False):
        imgs, labels = batch
        predictions, reconstruction = self(imgs)

        if prefix == "val":
            class_loss = F.cross_entropy(predictions, labels, weight=weight_val.cuda())
        elif prefix == "test":
            class_loss = F.cross_entropy(predictions, labels)

        rec_loss = F.mse_loss(reconstruction, imgs)
        loss = class_loss + self.hparams.alpha * rec_loss

        # at beginning of epoch
        if batch_idx == 0:

            # reset predictions from last epoch
            self.gt_val = []
            self.pr_val = []

            if plot:
                self.plot(imgs, reconstruction, prefix)

        # save labels and predictions for evaluation
        max_indices = torch.max(predictions, 1).indices
        self.gt_val += labels.tolist()
        self.pr_val += max_indices.tolist()

        logs = {f"{prefix}/loss": loss}
        return {f"{prefix}_loss": loss, "log": logs}

    def _shared_eval_epoch_end(self, outputs, prefix):

        print(f"\n---> metrics for entire {prefix} epoch are: \n")
        metrics = calc_metrics(self.gt_val, self.pr_val, verbose=True)
        avg_loss = torch.stack([x[f"{prefix}_loss"] for x in outputs]).mean()
        logs = {f"{prefix}/avg_loss": avg_loss}

        # tensorboard only saves scalars
        loggable_metrics = ["accuracy", "recall", "precision"]
        metrics = {f"{prefix}/{key}": metrics[key] for key in loggable_metrics}
        logs.update(metrics)

        return {f"{prefix}/avg_loss": avg_loss, "log": logs}


def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=hparams.log_name)
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create classifier and print summary
    model = Classifier(hparams)
    # summary(model, (hparams.nc, hparams.img_size, hparams.img_size), device="cpu")

    trainer = Trainer(
        logger=logger,
        gpus=hparams.gpus,
        max_epochs=hparams.max_epochs,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        weights_summary=None,
    )

    # retrieve COVIDx_v3 train dataset from COVID-Net paper
    covidx_train = COVIDx("train", hparams.data_root, hparams.dataset_dir)
    transform = Transform(MEAN.tolist(), STD.tolist(), hparams)

    if hparams.debug:
        plot_dataset(covidx_train)

    # k fold cross validation
    kfold = KFold(n_splits=hparams.folds)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(covidx_train)):
        print(f"training {fold} of {hparams.folds} folds ...")

        # split covidx_train further into train and val data
        train_ds = TransformableSubset(covidx_train, train_idx, transform=transform.train)
        val_ds = TransformableSubset(covidx_train, val_idx, transform=transform.test)

        # calc class weights of current folds
        global weight_train, weight_val
        weight_train = get_class_weights(covidx_train, train_idx)
        weight_val = get_class_weights(covidx_train, val_idx)

        if torch.cuda.is_available():
            weight_train.cuda()
            weight_val.cuda()

        train_dl = DataLoader(train_ds, batch_size=hparams.batch_size, num_workers=hparams.num_workers,)

        val_dl = DataLoader(val_ds, batch_size=hparams.batch_size, num_workers=hparams.num_workers,)

        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

    trainer.test(model)
    save_model(model, hparams.models_dir, hparams.log_name)
    print("done. have a good day!")


if __name__ == "__main__":

    args = parse_args()
    main(args)
