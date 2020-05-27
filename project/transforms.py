import torchvision.transforms as transforms


class Transform:
    """Defines transforms for training and testing
    """

    def __init__(self, mean, std, hparams):
        self.mean = mean
        self.std = std

        self.train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    hparams.image_size,
                    scale=(hparams.aug_min_scale, hparams.aug_max_scale),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=hparams.aug_rot),
                transforms.ColorJitter(
                    brightness=hparams.aug_bright, contrast=hparams.aug_cont
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.test = transforms.Compose(
            [
                transforms.Resize(hparams.image_size),
                transforms.CenterCrop(hparams.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )