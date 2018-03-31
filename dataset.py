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
    
        self.	
  
    def  __retrive__con tinue to find whtat tobe 