# 2018.3.29

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transforms import Scale
import os
from scipy.misc import imread, imsave, imresize, imshow
import scipy.io as sio
from PIL import Image

class smallNORB(Dataset):
    def __init__(self, root, split='train'):

        with open(os.path.join(root, split + '.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        self.root_dir = root
        self.img_dir = os.path.join(root, 'smallnorb_' + split)
        self.transform = transforms.Compose([
                                    # transforms.Scale(48, 48),
                                    Scale([48, 48]),
                                    transforms.CenterCrop([32, 32]),
                                    transforms.ToTensor()
                                    ])
        self.transform_aug = transforms.Compose([
                                    # transforms.Scale(48, 48),
                                    Scale([48, 48]),
                                    transforms.Pad(1),
                                    transforms.CenterCrop([32, 32]),
                                    transforms.RandomRotation(5),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                                    transforms.ToTensor()])

        self.if_aug = (split=='train')

        self.split = split


    def __getitem__(self, index):

        imgfile, label = self.data[index]

        img_path = os.path.join(self.root_dir, imgfile)
        # img = imread(img_path)
        # img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path)
        if self.if_aug:
            img = self.transform_aug(img)
        else:
            img = self.transform(img)
        # print (img) # 3,32,32
        return img, label

    def __len__(self):

        return len(self.data)


def collate_data(batch):

    imgs, labels = [], []
    batch_size = len(batch)

    for i, (img, label) in enumerate(batch):
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs), torch.LongTensor(np.array(labels))
