import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        # note that we do not convert to tensor here
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]
        self.transform = transform 

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset(transform=None)

first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)
first_data2 = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))