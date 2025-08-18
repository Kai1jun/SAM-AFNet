import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register
import open3d as o3d

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path, split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        # print('ImageFolder: self.repeat',self.repeat)
        # print('ImageFolder: self.cache', self.cache)
        # print('ImageFolder: self.size', self.size)

        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            print('pic_filename:',filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        # print('pic_idx:',idx)

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')


@register('pointcloud-folder')
class PointCloudFolder(Dataset):
    def __init__(self, label_path=None, split_file=None, split_key=None, first_k=None, cache='none', repeat=1):
        self.label_path = label_path  
        self.cache = cache
        self.repeat = repeat
        self.files = []

        if split_file is None:
            filenames = sorted(os.listdir(label_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]

        if first_k is not None:
            filenames = filenames[:first_k]

        for filename in filenames:
            label_file = os.path.join(label_path, filename)
            print('label_filename:',filename)
            self.append_file(label_file)

    def append_file(self, label_file):
        if self.cache == 'none':
            self.files.append(label_file)
        elif self.cache == 'in_memory':
            self.files.append(self.load_label(label_file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        label_file = self.files[idx % len(self.files)]
        # print('label_idx:',idx)
        if self.cache == 'none':
            label = self.load_label(label_file) if label_file else None
        elif self.cache == 'in_memory':
            label = label_file
        return label


    def load_label(self, label_file):
        
        data = np.loadtxt(label_file, delimiter=' ', usecols=range(6)) 
        
        return torch.tensor(data, dtype=torch.float32)


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs):
        # Received kwargs: {'cache': 'none', 'split_key': 'test'}
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
        self.dataset_3 = PointCloudFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx]

