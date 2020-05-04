from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from autoencoder import HealhtyAE

# normalization constants
MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

# other constants
AE_PATH = "./healthy_ae.pth"


class Classifier(pl.LightningModule):
    def __init__(self, hparams, autoencoder):
        super().__init__()
        self.hparams = hparams
        self.autoencoder = autoencoder

        # number of neurons in last dense layers
        self.nd = self.hparams.nf * 4 * (self.hparams.image_size // 4**3) ** 2

        self.classifier = nn.Sequential(
            # input (nc) x 256 x 256
            nn.Conv2d(self.hparams.nc, self.hparams.nf, 4, 2, 1),
            nn.BatchNorm2d(self.hparams.nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # input (nf) x 64 x 64
            nn.Conv2d(self.hparams.nf, self.hparams.nf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.hparams.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # input (nf*2) x 16 x 16
            nn.Conv2d(self.hparams.nf * 2, self.hparams.nf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.hparams.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            # input (nf*4) x 4 x 4
            nn.Flatten(),
            nn.Linear(self.nd, self.nd, bias=True),
            nn.Linear(self.nd, self.nd, bias=True),
            nn.Linear(self.nd, 3, bias=True),
            nn.Softmax(),
        )

    def forward(self, x):
        # create anomaly map
        decoded = self.autoencoder(x)
        anomaly = x - decoded

        # classify anomaly map
        pred = self.classifier(anomaly)
        return pred

    def prepare_data(self):

        transform = transforms.Compose(
            [
                transforms.Resize(self.hparams.image_size),
                transforms.CenterCrop(self.hparams.image_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN.tolist(), STD.tolist()),
            ]
        )

        dataset = ImageFolder(root=self.hparams.data_root + "/train", transform=transform)

        # split train and val
        end_train_idx = 7080

        self.train_dataset = Subset(dataset, range(0, end_train_idx))
        self.val_dataset = Subset(dataset, range(end_train_idx + 1, len(dataset)))
        self.test_dataset = ImageFolder(root=self.hparams.data_root + "/test", transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2)
        )

    def plot(self, x, r, a, prefix, n=16):
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
        plot = torch.tensor([], device=x1[0].device)

        for i in range(n):

            grid = vutils.make_grid([x[i], r[i], a[i]], 1)
            plot = torch.cat((plot, grid), 2)

            # add offset between image triplets
            if n > 1 and i < n - 1:
                border_width = 6
                border = torch.zeros(
                    plot.shape[0], plot.shape[1], border_width, device=x1[0].device
                )
                plot = torch.cat((plot, border), 2)

        name = f"{prefix}_input_anomaly_reconstr_images"
        self.logger.experiment.add_image(name, plot)

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def validation_epoch_end(self, outputs):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def test_epoch_end(self, outputs):
        return None


def main(hparams):
    logger = loggers.TensorBoardLogger(hparams.log_dir, prefix="classifier")

    # load pretrained autoencoder
    autoencoder = HealhtyAE()
    autoencoder.load_state_dict(torch.load(AE_PATH))
    autoencoder.eval()

    # create classifier
    model = Classifier(hparams, autoencoder)

    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.image_size, hparams.image_size), device="cpu")

    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=256, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=8, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nf", type=int, default=8, help="Number of feature maps in classifier")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)
