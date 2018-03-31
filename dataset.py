import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transform
import os
from scipy.misc import imread, imsave, imresize
import scipy.io as sio

class smallNORB(Dataset):
    def __init__(self, root, split='train'):

        with open('data/' + split + '.pkl', 'rb') as f:
            self.data = pickle.load(f)
    
        self.root_dir = root
        self.img_dir = os.path.join(root, 'smallnorb_' + split)
    
		self.transform = transforms.Compose([
									Scale(96, 96),
									transforoms.ToTensor()
									])
		self.transform_aug = transforms.Compose([
									Scale(96, 96),
									transforms.Pad(3),
									transforms.RandomCrop([96, 96]),
									transforoms.ToTensor()
									
		self.if_aug = (split==‘train’)

		self.split = split


	def __getitem__(self, index):

		imgfile, label = self.data[index]
		
		# dir_path = os.path.join(self.root_dir, ‘smallnorb_’ + self.split, label, label + )
		img_path = os.path.join(self.root_dir, imgfile)
		img = imread(img_path)
		if self.if_aug:
			img = self.transform_aug(img)
		else:
			img = self.transform(img)

		return img, label

	def __len__(self):

		return len(self.data)


	def collate_data(batch):
	
		imgs, labels = [], []
		batch_size = len(batch)

		for i, (img, label) in enumerate(batch):
			imgs.append(img)
			labels.append(label)
		return torch.stack(imgs)m torch.LongTensor(labels)
