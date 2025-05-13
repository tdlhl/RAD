import csv
import json
import logging
import os
import re
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from collections import defaultdict
from PIL import Image
import cv2
from dataset.randaugment import RandomAugment
from io import BytesIO
import torch
import torch.nn as nn
import nibabel as nib
from dataset.augmentation.augment import *

class Fair_ori_train_dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)

        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])
        self.text_list = np.asarray(data_info.iloc[:,1])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomGrayscale(),

                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
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

class ICD_Train_Dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)

        self.img_path_list = np.asarray(data_info.iloc[:,0])

        self.class_list = np.asarray(data_info.iloc[:,2:])
        
        self.text_list = np.asarray(data_info.iloc[:,1])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomGrayscale(),

                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])

    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):

            img_path = '/your_path/MIMIC-CXR-JPG/files/' + self.img_path_list[index]
            class_label = self.class_list[index]  
            caption_list = ''
            head = [
                "abscess of lung and mediastinum, pyothorax, mediastinitis",
                "acute bronchiolitis and unspecified acute lower respiratory infection",
                "bronchiectasis",
                "bronchitis",
                "chronic airway obstruction, not elsewhere classified",
                "complications and ill-defined descriptions of heart disease, cardiomegaly",
                "extrinsic asthma, unspecified",
                "farmers' lung and pneumoconiosis due to asbestos and mineral fibers",
                "flu due to avian flu virus",
                "fracture of rib(s), sternum and thoracic spine",
                "lung opacity",
                "other pleural conditions",
                "other respiratory disorders, consolidation",
                "pleurisy without mention of effusion or current tuberculosis",
                "pneumoconiosis",
                "pneumonia",
                "pneumonitis due to solids, liquids, and radiation",
                "pulmonary collapse",
                "pulmonary congestion, edema, and pleural effusion",
                "pulmonary emphysema and emphysematous bleb",
                "pulmonary fibrosis and other interstitial pulmonary diseases",
                "surgical instruments, materials and cardiovascular devices",
                "traumatic, spontaneous, and tension pneumothorax, air leak",
                "hypertension",
                "hyperosmolality and/or hypernatremia",
                "diabetes mellitus",
                "heart failure",
                "tachycardia",
                "chronic total occlusion of coronary artery",
                "hypotension",
                "myocardial infarction",
                "chronic ischemic heart disease",
                "atrial fibrillation and flutter",
                "primary pulmonary hypertension",
                "acute posthemorrhagic anemia",
                "endomyocardial fibrosis",
                "other coagulation defects",
                "malignant neoplasm of trachea",
                "atherosclerosis of aorta",
                "abnormal serum enzyme levels",
                "cardiomyopathy",
                "other cardiac arrhythmias",
                "vitamin D deficiency",
                "esophageal varices",
                "angina decubitus",
                "pulmonary embolism",
                "poisoning by, adverse effect of and underdosing of agents primarily affecting the cardiovascular system",
                "acute and subacute bacterial endocarditis",
                "polyarteritis nodosa",
                "arterial embolism and thrombosis",
                "acute and subacute endocarditis",
                "phlebitis and thrombophlebitis",
                "thalassemia",
            ]
            

            entity_details = self.text_list[index]  

            img = Image.open(img_path).convert('RGB') 
            image = self.transform(img)

            return {
            "image": image,
            "label": class_label,
            "label_dataset": 0,
            "caption": caption_list,
            "entity": entity_details
            }
    

class Skin_Train_Dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)

        self.img_path_list = np.asarray(data_info.iloc[:,0])

        self.class_list = np.asarray(data_info.iloc[:,2:])
        
        self.text_list = np.asarray(data_info.iloc[:,1])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomGrayscale(),

                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])

    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):

            img_path = '/your_path/skincap/' + self.img_path_list[index]
            class_label = self.class_list[index]  
            
            entity_details = self.text_list[index]  

            img = Image.open(img_path).convert('RGB') 
            image = self.transform(img)

            return {
            "image": image,
            "label": class_label,
            "label_dataset": 0,
            "entity": entity_details
            }

class NACC_Train_Dataset(Dataset):
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
