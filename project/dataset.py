import os
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image

class COVIDx(Dataset):

    def __init__(self, mode, root_dir="./data", transform=None):

        self.img_dir = os.path.join(root_dir, mode)
        self.mapping = {"normal": 0, "pneumonia": 1, "COVID-19": 2}
        self.transform = transform

        train_file = os.path.join(root_dir, "train_COVIDx3.txt")
        test_file = os.path.join(root_dir, "test_COVIDx3.txt")

        if mode == "train":
            self.paths, self.labels = self.read_file(train_file)
        elif mode == "test":
            self.paths, self.labels = self.read_file(test_file)
        else:
            Exception(f"Mode {mode} not supported.")

        # count number of images per class
        self.counter = Counter(self.labels)
        
        print(f"Class distribution in {mode} set: {self.counter}")
        print(f"Length of {mode} set: {len(self.paths)}")
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.img_dir, self.paths[idx])
        if not os.path.exists(img_name):
            Exception(f"Image {img_path} does not exist!")
            
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