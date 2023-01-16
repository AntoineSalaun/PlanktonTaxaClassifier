import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch 
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import RandomSampler, DataLoader, Subset, SubsetRandomSampler, RandomSampler

# Home-made loade is very convinient to handle the plankton data
class PlanktonLoader(Dataset):
    """Loads the plankton Classification dataset."""

    def __init__(self, csv_file, image_folder, unwanted_classes = None, transform=None):
        
        #Trying to delete unwanted files from the dataset
        try:
            for i in unwanted_classes:  
                shutil.rmtree(image_folder+i)
        except (FileNotFoundError):
            pass
        
        #print(csv_file)
        self.data_pre = pd.read_csv(csv_file) # data_pre is the .csv that contains the images names and correspondaing labels
        #print(self.data_pre)
        self.data = self.data_pre[~self.data_pre.taxon.str.contains('|'.join(unwanted_classes))] # we remove the unwanted class
        #self.data = self.data_pre
        self.data.index = range(len(self.data))         # reindexing after deletion
        print(' The data has a lenght of ', len(self.data))
        self.transform = transform # pre processing defined in experiment

        # First 2 columns contains the id for the image and the class of the image -> we make a dict from it
        self.dict = self.data.iloc[:,:2].to_dict()
        self.ids = self.dict["objid"] # identifier of each image
        print(' The id list has a lenght of ', len(self.ids)) # print the number of images
        self.classes = self.data["taxon"].unique() # List of unique class name

        # Assigns number to every class in the order which it appears in the data
        self.class_to_idx = {j: i for i, j in enumerate(self.classes)} 
        self.species = self.dict["taxon"] # list of taxa
        self.path_plankton = image_folder # Where the images are stored

        print('We have ', len(self.classes), 'classes')

    # defining the leng of the loader
    def __len__(self):
        return len(self.data)

    # how to get an item from the dataloader
    def __getitem__(self, idx):

        # make sure idx is an torch(int)
        if torch.is_tensor(idx):
            idx = idx.item()
            assert isinstance(idx, int)

        num = self.ids[idx] # Id of the indexed item
        loc = f"/{num}.jpg" # name of the file is id.jpg
        label = self.dict["taxon"][idx] # Find the label/class of the image at given index
        label_num = self.class_to_idx[label] # Convert it to int
        image = Image.open(self.path_plankton + self.dict["taxon"][idx] + loc) # find the image in the system
        if self.transform:
            image = self.transform(image) # apply the preprocessing

        return (image, label, label_num) # return image, label name and label number (each class is given a number)

    # build_loaders does the train/test split
    def build_loaders(dataset, sampling_factor, train_factor, batch_size, random_seed= 42 , shuffle_dataset = True  ): 
    
        num_samples = int(sampling_factor * len(dataset)) # we only take a portion of the data (unless sampling_factor = 1)
        train_size = int(train_factor * num_samples) # train_factor % of the data to be used for training
        test_size = num_samples - train_size # The remaining goes for testing

        # print what's up
        print('We use ', sampling_factor, ' of the data (',num_samples, ' samples) and the train factor is ', train_factor)
        print('Train set contains', train_size, 'images.')
        print('Test set contains', test_size, 'images.')

        # the data is shuffled and split wrt to the train_factor proportion
        train_dataset, test_dataset = random_split(Subset(dataset, np.random.permutation(np.arange(int(len(dataset))))[:train_size+test_size]), [train_size, test_size])

        # put in batches :
        trainloader_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        return trainloader_dataset, testloader_dataset # return the train and test loaders