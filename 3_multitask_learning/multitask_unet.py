# ORIGINAL CODE FROM https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulittaskUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(MulittaskUNet, self).__init__()

        self.features = init_features
        self.encoder1 = MulittaskUNet._block(in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = MulittaskUNet._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = MulittaskUNet._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = MulittaskUNet._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MulittaskUNet._block(self.features * 8, self.features * 16, name="bottleneck")

        # BEGIN MODIFICATION ALEXANDER KOENIG
        self.avg_pool = nn.AvgPool2d(kernel_size=14)
        self.fc1 = nn.Linear(self.features * 16, self.features * 32)
        self.fc2 = nn.Linear(self.features * 32, 3)
        self.softmax = nn.Softmax(dim=1)
        # END

        self.upconv4 = nn.ConvTranspose2d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = MulittaskUNet._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = MulittaskUNet._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = MulittaskUNet._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = MulittaskUNet._block(self.features * 2, self.features, name="dec1")

        self.conv = nn.Conv2d(in_channels=self.features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        # BEGIN MODIFICATION ALEXANDER KOENIG
        pooled = self.avg_pool(bottleneck)
        pooled = pooled.view(-1, self.features * 16)
        fc1 = self.fc1(pooled)
        fc2 = self.fc2(fc1)
        prediction = self.softmax(fc2)
        # END

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return prediction, torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
