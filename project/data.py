import os
from collections import Counter

import torch
from PIL import Image
from torch import randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset
from sklearn.utils import shuffle


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [
        TransformableSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


class TransformableSubset(Dataset):
    """
    Subset of a dataset at specified indices with transform.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (transform): transform on Subset 
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, labels

    def __len__(self):
        return len(self.indices)


class COVIDx(Dataset):
    def __init__(self, mode, data_root, dataset_dir, transform=None):

        self.img_dir = os.path.join(data_root, mode)
        self.mapping = {"normal": 0, "pneumonia": 1, "COVID-19": 2}
        self.transform = transform

        train_file = os.path.join(dataset_dir, "train_COVIDx3.txt")
        test_file = os.path.join(dataset_dir, "test_COVIDx3.txt")

        if mode == "train":
            self.paths, self.labels = self.read_file(train_file)

            # training set must be shuffled before cross validation
            self.paths, self.labels = shuffle(self.paths, self.labels)
        elif mode == "test":
            self.paths, self.labels = self.read_file(test_file)
        else:
            Exception(f"Mode {mode} not supported.")

        # count number of images per class
        self.counter = Counter(self.labels)

        # get targets in form [0, 1, 2, 1, ...]
        self.targets = [self.mapping[l] for l in self.labels]

        print(f"Class distribution in COVIDx {mode} set: {self.counter}")
        print(f"Length of COVIDx {mode} set: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.paths[idx])
        if not os.path.exists(img_name):
            Exception(f"Image {img_name} does not exist!")

        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.mapping[self.labels[idx]], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def read_file(self, file):
        paths, labels = [], []
        with open(file, "r") as f:
            lines = f.read().splitlines()

            for _, line in enumerate(lines):
                path, label = line.split(" ")[1:3]
                paths.append(path)
                labels.append(label)

        return paths, labels


class COVIDxNormal(Dataset):
    def __init__(self, mode, data_root, dataset_dir, transform=None):

        self.img_dir = os.path.join(data_root, mode)
        self.transform = transform

        train_file = os.path.join(dataset_dir, "train_COVIDx3.txt")
        test_file = os.path.join(dataset_dir, "test_COVIDx3.txt")

        if mode == "train":
            self.paths, self.labels = self.read_file(train_file)
        elif mode == "test":
            self.paths, self.labels = self.read_file(test_file)
        else:
            Exception(f"Mode {mode} not supported.")

        # count number of images per class
        self.counter = Counter(self.labels)

        print(f"Length of COVIDxNormal {mode} set: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.paths[idx])
        if not os.path.exists(img_name):
            Exception(f"Image {img_name} does not exist!")

        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(0, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def read_file(self, file):
        paths, labels = [], []
        with open(file, "r") as f:
            lines = f.read().splitlines()

            for _, line in enumerate(lines):
                path, label = line.split(" ")[1:3]
                if label == "normal":
                    paths.append(path)
                    labels.append(label)

        return paths, labels
