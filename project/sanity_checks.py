import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np

from unet import UNet
from utils import freeze
from transforms import Transform
from args import parse_args
from data import COVIDx
from autoencoder import NormalAE
from utils import scale_img_to_01

MEAN = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
STD = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)


def sanity_check_losses(dl, autoencoder):
    print("---STARTING TO SANITY CHECK LOSSES---")

    normal_loss = 0.0
    pneumonia_loss = 0.0
    covid_loss = 0.0

    normal_count = 0
    pneumonia_count = 0
    covid_count = 0

    # sanity check for check losses
    for idx, (orig, labels) in enumerate(dl):
        rec = autoencoder(orig)
        batch_size = len(labels)

        # iterate over batch and save individual losses
        for i in range(batch_size):
            if labels[i] == 0:
                normal_loss += F.mse_loss(orig[i], rec[i])
                normal_count += 1
            elif labels[i] == 1:
                pneumonia_loss += F.mse_loss(orig[i], rec[i])
                pneumonia_count += 1
            elif labels[i] == 2:
                covid_loss += F.mse_loss(orig[i], rec[i])
                covid_count += 1

    print("---RESULTS---")
    print(f"ran for {idx} iterations")
    print(f"normal count is \t {normal_count}")
    print(f"pneumonia count is \t {pneumonia_count}")
    print(f"covid count is \t\t {covid_count}")
    print(f"avg loss for normal images: \t {normal_loss/normal_count}")
    print(f"avg loss for pneumonia images: \t {pneumonia_loss/pneumonia_count}")
    print(f"avg loss for covid images: \t {covid_loss/covid_count}")


def plot_img_triplet(dl, plot_index, autoencoder):
    print("---STARTING TO LOG TRIPLE IMAGE---")

    # retrieve images
    orig, labels = iter(dl).next()

    # forward pass
    orig = orig[plot_index]
    rec = autoencoder(orig)[plot_index]
    anomaly = orig - rec
    scale_img_to_01(anomaly)

    # make grid and plot
    grid = utils.make_grid([orig, rec, anomaly])
    plot_tensor(grid)


def plot_all_anomaly_maps(dl, autoencoder):
    print("---STARTING TO PLOT ALL ANOMALY MAPS---")

    all_anomalies = torch.tensor([])
    # plotting all anomaly maps (without labels)
    for idx, (orig, labels) in enumerate(dl):
        rec = autoencoder(orig)
        anomaly = orig - rec
        for i in range(anomaly.shape[0]):
            scale_img_to_01(anomaly[i])

        all_anomalies = torch.cat([all_anomalies, anomaly], dim=0)

    grid = utils.make_grid(all_anomalies)
    plot_tensor(grid)


def plot_anomaly_maps_by_label(dl, autoencoder):
    print("---STARTING TO PLOT ANOMALY MAPS BY LABEL---")

    origs = torch.tensor([])
    normal_anomalies = torch.tensor([])
    pneumonia_anomalies = torch.tensor([])
    covid_anomalies = torch.tensor([])

    for idx, (orig, labels) in enumerate(dl):
        rec = autoencoder(orig)
        anomaly = orig - rec
        origs = torch.cat([origs, orig], dim=0)

        for i in range(anomaly.shape[0]):
            scale_img_to_01(anomaly[i])
            if labels[i] == 0:
                normal_anomalies = torch.cat([normal_anomalies, anomaly[i].unsqueeze(0)], dim=0)
            elif labels[i] == 1:
                pneumonia_anomalies = torch.cat([pneumonia_anomalies, anomaly[i].unsqueeze(0)], dim=0)
            elif labels[i] == 2:
                covid_anomalies = torch.cat([covid_anomalies, anomaly[i].unsqueeze(0)], dim=0)

    print("image order: \ntop row normal cases \nmiddle pneumonia \nbottom covid")
    all_imgs = torch.cat([origs, normal_anomalies, pneumonia_anomalies, covid_anomalies], dim=0)
    num_imgs_per_class = 100
    grid = utils.make_grid(all_imgs, nrow=num_imgs_per_class)
    plot_tensor(grid, save=True, save_pth="/Users/koenig/Desktop/plot_all_classes.png")


def plot_tensor(grid, save=False, save_pth="", dpi=2000):
    t = transforms.ToPILImage()
    arr = np.asarray(t(grid))

    fig = plt.figure(figsize=(10.0, 0.6), frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, aspect="auto")

    if save == True:
        fig.savefig(save_pth, dpi=dpi)
    plt.show()


def main(hparams):

    # load pretrained model
    autoencoder = NormalAE(hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.load_state_dict(torch.load(hparams.pretrained_ae_pth, map_location=device))
    autoencoder.eval()
    freeze(autoencoder)

    # load test dataset
    transform = Transform(MEAN.tolist(), STD.tolist(), hparams)
    covidx_test = COVIDx("test", hparams.data_root, hparams.dataset_dir, transform=transform.test,)
    dl = DataLoader(covidx_test, batch_size=hparams.batch_size, num_workers=hparams.num_workers,)

    # sanity check
    plot_anomaly_maps_by_label(dl, autoencoder)


if __name__ == "__main__":

    args = parse_args()
    main(args)
