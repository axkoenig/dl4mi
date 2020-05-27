import torchvision.transforms as transforms


class Transform:
    """Defines transforms for training and testing
    """

    def __init__(self, mean, std, hparams):
        self.mean = mean
        self.std = std
        self.hparams = hparams

    def train(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.hparams.image_size,
                    scale=(self.hparams.aug_min_scale, self.hparams.aug_max_scale),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=self.hparams.aug_rot),
                transforms.ColorJitter(
                    brightness=self.hparams.aug_bright, contrast=self.hparams.aug_cont
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def test(self):
        return transforms.Compose(
            [
                transforms.Resize(self.hparams.image_size),
                transforms.CenterCrop(self.hparams.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )