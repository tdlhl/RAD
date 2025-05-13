import csv
import json
import logging
import os
import random
import re
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import cv2
import pydicom
from skimage import exposure
from io import BytesIO
import torch
import torch.nn as nn
import nibabel as nib
from dataset.augmentation.augment import *

class Fair_ori_test_dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)

        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])
        self.text_list = np.asarray(data_info.iloc[:,1])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]) 

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = '/your_path/Harvard-FairVLMed/' + self.img_path_list[index]
        label = self.class_list[index]    # Fourth column is the label
        text = self.text_list[index]  
        npz_data = np.load(img_path)
        # Extract the image (assuming the key is 'slo_fundus')
        image = npz_data['slo_fundus'].astype(np.float32)
        # Normalize the image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "label_dataset": 0,
            "entity": text
        }

class ICD_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.text_list = np.asarray(data_info.iloc[:,1])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.class_list = np.asarray(data_info.iloc[:,2:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):

        img_path = '/your_path/MIMIC-CXR-JPG/files/' + self.img_path_list[index]
        class_label = self.class_list[index] 
        entity_details = self.text_list[index]  

        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label,
            "entity": entity_details
            }
    
    def __len__(self):
        return len(self.img_path_list)

class Skin_Test_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.text_list = np.asarray(data_info.iloc[:,1])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.class_list = np.asarray(data_info.iloc[:,2:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):

        img_path = '/your_path/skincap/' + self.img_path_list[index]
        class_label = self.class_list[index] 
        entity_details = self.text_list[index]  

        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label,
            "entity": entity_details
            }
    
    def __len__(self):
        return len(self.img_path_list)
 
class NACC_Test_Dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)

        self.img_path_list = np.asarray(data_info.iloc[:,0])

        self.class_list = np.asarray(data_info.iloc[:,2:])
        
        self.text_list = np.asarray(data_info.iloc[:,1])

        self.augmentation = False

    
    def __len__(self):
        return len(self.img_path_list)

    def _augment(self, img_data):
        return img_data
    
    def __getitem__(self, index):

        img_path = '/your_path/NACC_images/' + self.img_path_list[index]
        class_label = self.class_list[index]  
            
        entity_details = self.text_list[index]  

        # img = Image.open(img_path).convert('RGB') 
        # image = self.transform(img)
        img_data = nib.load(img_path).get_fdata()
        if img_data.ndim > 3:
            img_data = img_data[:, :, :, 0]
        img_data = nnUNet_resample_and_normalize(img_data, [96, 96, 96], is_seg=False)
        img_data = img_data.transpose([2, 1, 0])
        img_data = img_data[np.newaxis, :]

        if self.augmentation:
            img_data = self._augment(img_data)

        img_data = torch.from_numpy(img_data).float()

        return {
            "image": img_data,
            "label": class_label,
            "label_dataset": 0,
            "entity": entity_details
        }
