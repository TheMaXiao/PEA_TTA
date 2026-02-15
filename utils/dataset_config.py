from torchvision import transforms
from utils.functions import load_txt
from torchvision import datasets
from PIL import Image
import torchvision
import os
import numpy as np

standard_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
standard_transform_vit = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    normalize
])



# domains_info = ['snow', 'fog', 'frost', 'brightness', 'impulse_noise', 'gaussian_noise', 'shot_noise', 'zoom_blur','contrast', 'defocus_blur','elastic_transform', 'gaussian_blur', 'jpeg_compression','motion_blur','pixelate'
#                 , 'saturate', 'spatter', 'speckle_noise','glass_blur', 'near_focus', 'far_focus' ,'fog_3d' ,'flash' ,'color_quant', 'low_light']


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str, severity :int,
                 transform=None, target_transform=None):
        # assert name in corruptions
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

def IMAGENETC(root :str, name :str, severity :int, transform=None):
    # assert name in domains_info
    data_path = os.path.join(root, name, str(severity))
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    return dataset

def load_datasets_for_domains(domains_list, num_severity, root_path, transform, source_set=None):
    # assert name in domains_info
    datasets = [[] for domain in range(len(domains_list))]
    
    for domain in range(len(domains_list)):
        if domains_list[domain] == 'source':
            for i in range(num_severity):
                datasets[domain].append(source_set)
        else:
            for i in range(num_severity):
                datasets[domain].append(IMAGENETC(root_path, domains_list[domain], 5, transform))
    return datasets
