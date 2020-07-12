import torchvision.transforms as transforms


class Transform:
    """Defines transforms for training and testing
    """

    def __init__(self, mean, std, hparams):
        self.train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=hparams.aug_rot, expand=True),
                transforms.RandomResizedCrop(
                    hparams.img_size, scale=(hparams.aug_min_scale, hparams.aug_max_scale),
                ),
                transforms.ColorJitter(brightness=hparams.aug_bright, contrast=hparams.aug_cont),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.test = transforms.Compose(
            [
                transforms.Resize(hparams.img_size),
                transforms.CenterCrop(hparams.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
