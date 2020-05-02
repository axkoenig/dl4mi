nc = 3                    # Number of channels in the training images
nz = 512                  # Size of z latent vector (size of generator input)
nfe = 32                  # Size of feature maps in generator
nfd = 32                  # Size of feature maps in discriminator

class HealhtyAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Input Dimensions (3 x 224 x 224), Output Dimensions (2x2x1024)
        self.encoder = nn.Sequential(
            # input (nc) x 256 x 256
            nn.Conv2d(nc, nfe, 4, 2, 1),
            nn.BatchNorm2d(nfe),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe) x 128 x 128
            nn.Conv2d(nfe, nfe*2, 4, 2, 1),
            nn.BatchNorm2d(nfe*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*2) x 64 x 64
            nn.Conv2d(nfe*2, nfe*4, 4, 2, 1),
            nn.BatchNorm2d(nfe*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*4) x 32 x 32
            nn.Conv2d(nfe*4, nfe*8, 4, 2, 1),
            nn.BatchNorm2d(nfe*8),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*8) x 16 x 16
            nn.Conv2d(nfe*8, nfe*16, 4, 2, 1),
            nn.BatchNorm2d(nfe*16),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*16) x 8 x 8
            nn.Conv2d(nfe*16, nfe*32, 4, 2, 1),
            nn.BatchNorm2d(nfe*32),
            nn.LeakyReLU(0.2, inplace=True),
            # input (nfe*32) x 4 x 4, i.e. 2048 x 4 x 4
            nn.Conv2d(nfe*32, nz, 4, 2, 1),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(0.2, inplace=True),
            # output 2x2x512
        )

        self.decoder = nn.Sequential(             
            # input (nz) x 2 x 2, i.e. 512 x 2 x 2 in our case
            nn.ConvTranspose2d(nz, nfd * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfd * 32),
            nn.ReLU(True),

            # input (nfd*32) x 4 x 4
            nn.ConvTranspose2d(nfd*32, nfd * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfd * 16),
            nn.ReLU(True),

            # input (nfd*16) x 8 x 8
            nn.ConvTranspose2d(nfd * 16, nfd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 8),
            nn.ReLU(True),

            # input (nfd*8) x 16 x 16
            nn.ConvTranspose2d(nfd * 8, nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 4),
            nn.ReLU(True),

            # input (nfd*4) x 32 x 32
            nn.ConvTranspose2d(nfd * 4, nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 2),
            nn.ReLU(True),

            # input (nfd*2) x 64 x 64
            nn.ConvTranspose2d(nfd * 2, nfd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd),
            nn.ReLU(True),

            # input (nfd) x 128 x 128
            nn.ConvTranspose2d(nfd, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output (nc) x 256 x 256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x