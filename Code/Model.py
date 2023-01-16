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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from Plot import *
from Net import DFL_VGG16


# The Image Classification class is the model
# it contains all the methods related to training and evaluating the model
class ImageClassificationBase(nn.Module):
    
    def __init__(self,
                    optimizer:torch.optim,
                    wd,
                    criterion:callable,
                    network,
                    learning_rate,
                    num_epochs
                    ) -> None:
        super(ImageClassificationBase, self).__init__() # useful PyTorch super intialization

        self.criterion = criterion
        self.lr = learning_rate
        self.epochs = num_epochs
        self.wd = wd

        # self.device is lets us use a GPU if one is usable
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('using cuda')
        else:
            self.device = torch.device("cpu")
            print('Not using cuda')
        
        self.net = network.to(self.device) # The model has a network
        self.optimizer = optimizer(self.net.parameters(),self.lr, weight_decay = self.wd)



    def training_step(self, batch): #computes the loss of a training batch
        self.net.train()
        self.net.to(self.device) 

        images, label_num = batch[0].to(self.device) ,  batch[2].to(self.device) 

        out = self.net(images).cpu()            # Generate predictions
        loss = self.criterion(out, label_num.cpu()) # Compute loss
        return loss
    
    def validation_step(self, batch): #computes the accuracy and loss of a validation batch
        self.net.to(self.device)
        self.net.eval() #no grad

        images, label_num = batch[0].to(self.device) ,  batch[2].to(self.device)
        out = self.net(images).cpu()                    # Generate predictions
        loss = self.criterion(out, label_num.cpu())   # Calculate loss
        acc = Plot.accuracy(out, label_num.cpu())           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):  #computes the val accuracy and val loss of an epoch from the val batches
        self.net.eval()

        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()     # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]      
        epoch_acc = torch.stack(batch_accs).mean()        # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result, t0): #prints the results at the end of each epoch
        self.net.eval()

        tf = time.time() # used to compute the computation time of an epcoh (thanks to t0 in arguments)
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f} - computation time (min): {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], (tf-t0)/60))
    
    

    @torch.no_grad()
    def evaluate(self, val_loader): #run the validation_step on each batch of an epoch and prints the epoch's results. 
        self.net.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)

    def fit(self, train_loader, val_loader): #train methods.
        print('starts training')
        history = [] # the histroy of results is stored in a list
        
        for epoch in range(self.epochs): # for each epoch
            t0 = time.time() #start-time (used to calculate the computation time)
            self.net.train() # PyTorch train mode
            train_losses = []

            for batch in train_loader: #for each train batch 
                loss = self.training_step(batch) # compute the loss of the batch
                train_losses.append(loss) # save the loss
                loss.backward() # gradient descent, learn
                self.optimizer.step() # update weights
                self.optimizer.zero_grad()
                
            result = self.evaluate(val_loader) # asses the results on the validation set
            result['train_loss'] = torch.stack(train_losses).mean().item() # record the train loss
            self.epoch_end(epoch, result, t0) # print results of the evaluation on the test set
            history.append(result) # record thes results in history
                
        return history

        