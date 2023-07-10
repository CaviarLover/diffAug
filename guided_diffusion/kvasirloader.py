import torch
import torch.nn
import numpy as np
import os
import os.path 
import cv2
from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, image_size=256, transform=None):
        '''
        directory is expected to contain some folder structure:
                  -images
                    -1.png
                    -2.png
                    ...
                  -masks
                    -1.png
                    -2.png
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.img_dir = os.path.join(directory, "images")
        self.mask_dir = os.path.join(directory, "masks") 
        
        self.database = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.database = sorted(self.database)

        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size))])
        else:
            self.transform = transform

    def __getitem__(self, x):
        
        file = self.database[x]

        img = Image.open(os.path.join(self.img_dir, file))
        img = img.convert('RGB')
        label = self.binary_loader(os.path.join(self.mask_dir, file))
        
        img = self.transform(img)
        label = self.transform(label)
        
        return (img, label)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = ImageOps.invert(img)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return len(self.database)


class KvasirTestset:
    def __init__(self, directory, testsize=256):
        self.testsize = testsize
        self.directory = os.path.expanduser(directory)
        self.img_dir = os.path.join(directory, "images")
        self.mask_dir = os.path.join(directory, "masks") 
        
        self.database = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.database = sorted(self.database)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.testsize, self.testsize))])

            #transforms.Normalize([0.485, 0.456, 0.406],
            #                     [0.229, 0.224, 0.225])])
        #self.gt_transform = transforms.ToTensor()
        self.size = len(self.database)
        self.index = 0

    def load_data(self):
        
        file = self.database[self.index]

        #image = self.rgb_loader(os.path.join(self.img_dir, file))
        #image = self.transform(image).unsqueeze(0)
        #gt = self.binary_loader(os.path.join(self.mask_dir, file))
        #gt = self.gt_transform(gt)

        image = cv2.imread(os.path.join(self.img_dir, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        gt = cv2.imread(os.path.join(self.mask_dir, file), 0)
        gt = self.transform(gt)
        gt = torch.where(gt > 0.5, 1, 0)[None, ...].float()

        name = file[:-4] 
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
