import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets

import random



def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

corruptions = load_txt('./corruptions.txt')    

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str, severity :int,
                 transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
    
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.data = self.data[0+10000*(severity-1):10000+10000*(severity-1)]
        self.targets = self.targets[0+10000*(severity-1):10000+10000*(severity-1)]
        # print(self.data.shape)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)