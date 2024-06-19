from args import *
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms


mapping_path = 'imagenet-object-localization-challenge/LOC_synset_mapping.txt'
train_path = 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train'
val_path = 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val'
val_solution_path = '/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv'

# Define the transform for training
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(256/480, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: x[:3, :, :]),  # Remove alpha channel
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.Normalize(mean=mean, std=std),
])

# Define the transform for validation (central crop)
valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=mean, std=std),
])

## Datasets and DataLoaders

class INDataset(Dataset):

        def __init__(self, image_ids, labels, target_directory, transform=None):

            self.transform = transform
            self.image_ids = image_ids
            self.labels = labels
            self.target_directory = target_directory

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, index):

            img = Image.open(os.path.join(self.target_directory, self.image_ids[index]))
            img = np.array(img)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            label = torch.tensor(self.labels[index], dtype=torch.long)

            if self.transform:
                return self.transform(img), label
            else:
                return img, label

def get_mappings(mapping_path):

    # Creating of mapping dictionaries to get the image classes
    class_mapping_dict = {}             #   {'n01440764': 'tench, Tinca tinca', ..}
    class_mapping_dict_number = {}      #   {0: 'tench, Tinca tinca',...}
    mapping_class_to_number = {}        #   {'n01440764': 0,...}
    mapping_number_to_class = {}        #   {0: 'n01440764',...}


    for i, line in enumerate(open(mapping_path)):
        class_mapping_dict[line[:9].strip()] = line[9:].strip()
        class_mapping_dict_number[i] = line[9:].strip()
        mapping_class_to_number[line[:9].strip()] = i
        mapping_number_to_class[i] = line[:9].strip()
    
    return mapping_class_to_number

def get_dataframes():
    mapping_class_to_number = get_mappings(mapping_path)

    # Creation of training image ids and labels

    train_labels = []
    train_ids = []
    for train_class in tqdm(os.listdir(train_path)):
        for i, image_id in enumerate(os.listdir(train_path + '/' + train_class)):
                path =  train_class + '/' + image_id
                train_ids.append(path)
                true_class = mapping_class_to_number[path.split('/')[-2]]
                train_labels.append(true_class)


    train_ids = np.array(train_ids)
    train_labels = np.array(train_labels)

    train_df = pd.DataFrame({'image_id': train_ids, 'label': train_labels})

    ## Validation data

    val_df = pd.read_csv(val_solution_path)
    val_df.rename(columns={'ImageId': 'image_id', 'PredictionString': 'label'}, inplace=True)
    val_df['label'] = val_df['label'].apply(lambda x: x.split()[0])
    val_df['label'] = val_df['label'].apply(lambda x: mapping_class_to_number[x])
    val_df['image_id'] = val_df['image_id'].apply(lambda x: x+'.JPEG')

    return train_df, val_df

     

def get_dataloaders(train_df, val_df,batch_size):

    train_dataset = INDataset(image_ids=train_df.image_id, labels=train_df.label, target_directory=train_path, transform=train_transforms)
    val_dataset = INDataset(image_ids=val_df.image_id, labels=val_df.label, target_directory=val_path, transform=valid_transforms)

    train_dl = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

    val_dl = DataLoader(val_dataset, batch_size= batch_size, shuffle=True)

    return train_dl, val_dl

