import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from dataloader.augmentation import color_augumentor
from dataloader.brightness_adjust import gamma_correction, add_spot_light
import random

ROOT = '/data/CelebA/CelebA_Spoof_Face/'

class Image_Loader(Dataset):
    def __init__(self, root_path='./data_train.csv', image_size=[224, 224], transforms_data=True, aug=False, phase='train'):
        
        self.data_path = pd.read_csv(root_path)
        self.image_size = image_size
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        self.aug = aug
        self.phase = phase 
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, item):

        # load 
        image_path = ROOT + os.path.join(self.data_path.iloc[item, 0])
        image = Image.open(image_path)

        if self.aug==True:

            gamma_correction_flag = random.choice([True, False])

            if gamma_correction_flag==True:
                image = gamma_correction(image)


        # live or spoof
        # spoof_label = []
        # spoof_label.append(self.data_path.iloc[item, 44])
        # spoof_label = torch.from_numpy(np.array(spoof_label, dtype=np.float32))
        spoof_label = self.data_path.iloc[item, 44]
        spoof_label = torch.from_numpy(np.array(spoof_label, dtype=np.float32))


        # 40 face attributes
        # atr_label = []
        # atr_label.append(self.data_path.iloc[item, 1:41])
        # atr_label = torch.from_numpy(np.array(atr_label, dtype=np.float32))
        atr_label = self.data_path.iloc[item, 1:41]
        atr_label = torch.from_numpy(np.array(atr_label, dtype=np.float32))

        # spoof type live 0, photo 1, poster 2, a4 3, upper body mask 5,...
        # spoof_type_label = []
        # spoof_type_label.append(self.data_path.iloc[item, 41])
        # spoof_type_label = torch.from_numpy(np.array(spoof_type_label, dtype=np.float32))
        spoof_type_label = self.data_path.iloc[item, 41]
        spoof_type_label = torch.from_numpy(np.array(spoof_type_label, dtype=np.float32))

        # illuminations
        # illum_label = []
        # illum_label.append(self.data_path.iloc[item, 42])
        # illum_label = torch.from_numpy(np.array(illum_label, dtype=np.float32))
        illum_label = self.data_path.iloc[item, 42]
        illum_label = torch.from_numpy(np.array(illum_label, dtype=np.float32))

        # Environments: live, indoor, outdoor
        # env_label = []
        # env_label.append(self.data_path.iloc[item, 43])
        # env_label = torch.from_numpy(np.array(env_label, dtype=np.float32))
        env_label = self.data_path.iloc[item, 43]
        env_label = torch.from_numpy(np.array(env_label, dtype=np.float32))

        if self.transforms_data == True and self.phase=='train':
            data_transform = self.transform(True, True, False, True)
            image = data_transform(image)
        else:
            data_transform = self.transform_test()
            image = data_transform(image)
        
        # print(image.size())

        return image, atr_label, spoof_type_label, illum_label, env_label, spoof_label

    def transform(self, horizon_flip, vertitcal_flip, rotation, totensor):
        options = []

        if True:
            options.append(transforms.Resize(self.image_size))
        if horizon_flip:
            options.append(transforms.RandomHorizontalFlip(p=0.5))
        if vertitcal_flip:
            options.append(transforms.RandomVerticalFlip(p=0.5))
        if rotation:
            options.append(transforms.RandomRotation(30, resample=False, expand=False, center=None, fill=None))
        if totensor:
            options.append(transforms.ToTensor())

        transform = transforms.Compose(options)

        return transform

    def transform_test(self):
        options = []

        options.append(transforms.Resize(self.image_size))
        options.append(transforms.ToTensor())

        transform = transforms.Compose(options)

        return transform

if __name__ == '__main__':

    dataset = Image_Loader(root_path='./data_train_all.csv', image_size=[128, 128], transforms_data=True)
    data = iter(dataset)
    import ipdb;ipdb.set_trace()
    print(len(dataset))
    # for i in range(len(dataset)):
    #     img, label, path = next(data)
    #     if img.size(0)==0 or img.size(1)==0 or img.size(2)==0:
    #         print(path)