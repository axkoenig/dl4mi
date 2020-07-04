from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing
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
from torchvision import models
from sklearn.model_selection import KFold

from autoencoder import NormalAE
from data import COVIDx, TransformableSubset
from transforms import Transform
from utils import calc_metrics, freeze


# normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


class Classifier(pl.LightningModule):
    def __init__(self, hparams, autoencoder):
        super().__init__()
        self.hparams = hparams
        self.autoencoder = autoencoder

        # variables to save model predictions
        self.gt_train = []
        self.pr_train = []
        self.gt_val = []
        self.pr_val = []

        # freeze resnet50 
        self.classifier = models.resnet50(pretrained=True)
        self.classifier.eval()
        freeze(self.classifier)

        # replace fc layers
        num_features = self.classifier.fc.in_features
        num_classes = 3
        self.classifier.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # TODO revert!
        # create anomaly map
        # reconstructed = self.autoencoder(x)
        reconstructed = x
        anomaly = x - reconstructed

        # classify anomaly map
        prediction = self.classifier(x)
        prediction = F.softmax(prediction)

        return {"reconstructed": reconstructed, "anomaly": anomaly, "prediction": prediction}

    def test_dataloader(self):
        transform = Transform(MEAN.tolist(), STD.tolist(), self.hparams)
        covidx_test = COVIDx("test", self.hparams.data_root, self.hparams.dataset_dir, transform=transform.test)

        return DataLoader(
            covidx_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
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
        a = [denormalization(i) for i in a[:n]]

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
        imgs, labels = batch
        out = self(imgs)
        predictions = out["prediction"]
        loss = F.cross_entropy(predictions, labels)

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
        
        print(f"---> metrics for entire train epoch are: \n")
        metrics = calc_metrics(self.gt_train, self.pr_train, verbose=True)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {f"train/avg_loss": avg_loss}
        
        # tensorboard only saves scalars
        loggable_metrics = ["accuracy", "recall", "precision"]
        metrics = {f"train/{key}": metrics[key] for key in loggable_metrics}
        logs.update(metrics)

        return {"train/avg_loss": avg_loss, "log": logs}

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
        predictions = out["prediction"]
        loss = F.cross_entropy(predictions, labels)

        # at beginning of epoch
        if batch_idx == 0:

            # reset predictions from last epoch
            self.gt_val = []
            self.pr_val = []
            
            if plot:
                self.plot(imgs, out["reconstructed"], out["anomaly"], prefix)
            
        # save labels and predictions for evaluation
        max_indices = torch.max(predictions, 1).indices
        self.gt_val += labels.tolist()
        self.pr_val += max_indices.tolist()

        logs = {f"{prefix}/loss": loss}
        return {f"{prefix}_loss": loss, "log": logs}

    def _shared_eval_epoch_end(self, outputs, prefix):
        
        print(f"---> metrics for entire {prefix} epoch are: \n")        
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
    
    # load pretrained autoencoder
    autoencoder = NormalAE(hparams)
    autoencoder.load_state_dict(torch.load(hparams.ae_pth, map_location=torch.device("cpu")))
    autoencoder.eval()
    freeze(autoencoder)

    # create classifier
    model = Classifier(hparams, autoencoder)

    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.img_size, hparams.img_size), device="cpu")
    
    # retrieve COVIDx_v3 dataset from COVID-Net paper
    covidx_train = COVIDx("train", hparams.data_root, hparams.dataset_dir)
    transform = Transform(MEAN.tolist(), STD.tolist(), hparams)

    if hparams.debug:
        plot_dataset(covidx_train)

    # k fold cross validation
    kfold = KFold(n_splits=hparams.folds)
    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs, nb_sanity_val_steps=hparams.nb_sanity_val_steps)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(covidx_train)):
        print(f"Training {fold} of {hparams.folds} folds ...")

        # split covidx_train further into train and val data 
        train_ds = TransformableSubset(covidx_train, train_idx, transform=transform.train)
        val_ds = TransformableSubset(covidx_train, valid_idx, transform=transform.test)
        
        # get labels in subset for rebalancing
        train_labels = [covidx_train.targets[i] for i in train_idx]

        # configure sampler to rebalance training set
        weights = 1 / torch.Tensor(
            [
                train_labels.count(0),
                train_labels.count(1),
                train_labels.count(2),
            ]
        )
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, hparams.batch_size)

        train_dl = DataLoader(
            train_ds,
            batch_size=hparams.batch_size,
            sampler=sampler,
            num_workers=hparams.num_workers,
        )

        val_dl = DataLoader(
            val_ds, batch_size=hparams.batch_size, num_workers=hparams.num_workers,
        )
        
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)
    
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser("Trains a classifier for COVID-19 detection")

    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Dataset root directory with txt files")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--log_name", type=str, default="classifier", help="Name of logging session")
    parser.add_argument("--ae_pth", type=str, default="models/autoencoder_30_05_2020_16_09_52_bs16_ep40_tl0.0064.pth", help="Path of trained autoencoder")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--img_size", type=int, default=224, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=4, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nfc", type=int, default=8, help="Number of feature maps in classifier")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    parser.add_argument("--nz", type=int, default=1024, help="Autoencoder param - Size of latent code")
    parser.add_argument("--nfe", type=int, default=32, help="Autoencoder param - Number of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Autoencoder param - Number of feature maps in decoder")
    parser.add_argument("--aug_min_scale", type=float, default=0.75, help="Minimum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_max_scale", type=float, default=1.0, help="Maximum scale arg for RandomResizedCrop")
    parser.add_argument("--aug_rot", type=float, default=5, help="Degrees arg for RandomRotation")
    parser.add_argument("--aug_bright", type=float, default=0.2, help="Brightness arg for ColorJitter")
    parser.add_argument("--aug_cont", type=float, default=0.1, help="Contrast arg for ColorJitter")
    parser.add_argument("--folds", type=int, default=10, help="How many folds to use for cross validation")
    parser.add_argument("--nb_sanity_val_steps", type=int, default=0, help="Number of sanity val steps")
    
    args = parser.parse_args()
    main(args)

# TODO CHECK IF WE ARE ACTUALLY PLOTTING THE CORRECT METRICS

# TODO run resnet on image directly 
# TODO run resnet on anomaly map
# TODO increase capability of autoencoder because full variability of healthy data needs to be captured! 
# TODO use Unet as autoencoder -> https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
# TODO why is my loss stagnating (run resnet classifier on image directly, should reduce loss)
# TODO maybe reduce learning rate
# TODO check if kfold can shuffle directly
# TODO fix training accuracy metrics 
# TODO think about loss function
# TODO maybe introduce thresholing to anomaly map 
# TODO why is the anomaly map not mostly black? visualise without normalization 
# TODO research if loss function weighting or rather dataset rebalancing (nll_loss = nn.CrossEntropyLoss(weight=torch.tensor([2., 2., 250.]).to('cuda')))
# TODO check if makes sense to use torch.abs(anomaly_map) https://github.com/chirag126/CoroNet/blob/728049e695c4efe0a11dc2a1282dc4f16af504f4/train_CIN.py#L91
# TODO check if images contain same info on all three channels or not
# TODO add test/accuracy --> tensorboard will format it nicely

# Differentiate from CoroNet
# - maybe use two autoencoders (normal, pneumonia) as feature extractors (use only encoder) and then classify directly & fine tune encoders jointly
# - check if SVM improves performance
