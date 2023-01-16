from torch.utils.data import Dataset, DataLoader, random_split
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
import os
import datetime

# the class Plot contains all the methods related to print/writing relusts
class Plot():
    # subplot random lets us print 16 random images of the dataset. 
    def subplot_random(trainloader_dataset, saving_location):
        im, lab, lab_num = next(iter(trainloader_dataset)) # iterate through the train data
        fig=plt.figure(figsize=(15, 15)) # construct figure

        for idx,(i,j) in enumerate(zip(im,lab)): # in the 16 images
            ax = fig.add_subplot(4,4,idx) # subplot
            print(i.squeeze().numpy()) # number images
            ax.imshow(i.squeeze().numpy())        # show images
            ax.set_title(j) # give the label as titles to the image
        plt.savefig(saving_location) # save image mosaic
        plt.close()

    # it prints 16 random images with its associated prediction s 
    def plot_random_output(testloader_dataset, dataset, model, saving_location):
        images, labels, label_num = next(iter(testloader_dataset)).to(model.device) #get the images
        fig=plt.figure(figsize=(15, 15)) # build figure
        _, preds = torch.max(model.net(images).cpu(), dim=1) # compute top1 prediction from output
        
        count = 0
        for idx,(i,j,k) in enumerate(zip(images, labels, preds)): # goes throug the images
            idx += 1
            ax = fig.add_subplot(4,4,idx)
            ax.imshow(i.squeeze().numpy())          # print them
            title = j + ' -> ' + dataset.classes[k] # get the ground-truth label and the model prediction
            ax.set_title(title) # use this as a title
            if j == dataset.classes[ k] :
                count = count +1
        plt.savefig(saving_location) # save the mosaic
        plt.close()

        print(count, 'good predictions. Accuracy : ', count/len(preds) ) # print the accuracy on this batch

    # compute accuracy
    def accuracy(outputs, labels): 
        _, preds = torch.max(outputs, dim=1) # max the outpu to make a top1 prediction
        return torch.tensor(torch.sum(preds == labels).item() / len(preds)) # count the number of good predictions 
    
    # compute accuracy from history 
    def plot_accuracies(history, num_epochs, saving_location_g,saving_location_d):
        accuracies = [x['val_acc'] for x in history]  # get the accuracies from history (after the training is done)
        pd.DataFrame(np.column_stack([ accuracies]), columns=['accuracies']).to_csv(saving_location_d)  # save them as a .csv 
        
        plt.plot(accuracies, '-x') # plot it
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.savefig(saving_location_g) # save the polot
        plt.close()

    # plot train and test loss
    def plot_losses(history,num_epochs,saving_location_g, saving_location_d):
        train_losses = [x.get('train_loss') for x in history] # get the train losses from history (after the training is done)
        val_losses = [x['val_loss'] for x in history] # get the trvalidationain losses from history (after the training is done)
        # save it as .csv for later plots
        pd.DataFrame(np.column_stack([ train_losses,val_losses ]), columns=['train loss', 'val loss']).to_csv(saving_location_d) 

        plt.plot(train_losses, '-bx') # plot
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.savefig(saving_location_g) # save
        plt.close()

    # compute the successes and prediction per class in a batch
    def class_success(outputs, label_num):
        _, preds = torch.max(outputs.cpu(), dim=1) # top1 prediction 
        tries, successes = np.zeros(84), np.zeros(84) # initialize arrays of tries and success

        for i in range(len(preds)): # for each prediction
            tries[label_num] = tries[label_num] + 1 # count it as a try for this class
            if preds[i] == label_num[i]: # if the prediction matches the label
                successes[label_num] = successes[label_num] + 1 # count it as a success on this class
        return successes, tries # return the number of succes and tries per class

    # compute and print the accuracies per class over the whole data
    def class_accuracies(self, model, dataset, val_loader, saving_location):
        pred_per_class, correct_pred_per_class = np.zeros(84), np.zeros(84) # initialize
        accuracy_per_class = np.zeros(len(pred_per_class))

        for batch in val_loader: # for each validation batch 
            # compute the number of predictions and correct predictions per class
            successes, passes = self.class_success(model.net(batch[0].to(model.device)).cpu() , batch[2].cpu()) 
            # accumulate them over the batch
            correct_pred_per_class, pred_per_class = correct_pred_per_class + successes, pred_per_class + passes   

        
        for i in range(len(pred_per_class)):  # for each class 
            if pred_per_class[i] == 0: # if there was no prediction
                print('Class #', i, ' ', dataset.classes[i] , 'was never trained on') # print it
            else: # if there was at least ione prediction
                accuracy_per_class[i] = correct_pred_per_class[i] / pred_per_class[i] # compute the accuracy for each class
                print('Class #', i, ' ', dataset.classes[i] , 'trained on', int(pred_per_class[i]) ,'times -> accuracy :', accuracy_per_class[i]) # print it

        # save it
        pd.DataFrame(np.column_stack([dataset.classes, pred_per_class, accuracy_per_class]), columns=['Class', '# predictions ', 'Accuracy']).to_csv(saving_location)

    
    # create a new folder to save the experiment results
    def new_folder(saving_location):
        # the name of the folder is unique because it contains the date
        folder_name = saving_location+'Experiment_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
        os.makedirs(folder_name) # create folder
        print('new folder : ', folder_name) # lets us link a log with an experiment folder
        return folder_name
    
    # create a file that contains the hyper parameter of the experiment
    def write_param(exp_name, wd, batch_size, sampling_factor,train_factor,num_epochs,lr,opt_func,crit):
        with open(exp_name+'/parameters.txt', 'w') as f:
            f.write("sampling_factor = " + str(sampling_factor) + "\n" 
            + "train_set_proportion = " + str(train_factor) + "\n" 
            + "num_epochs = " + str(num_epochs) + "\n" 
            + "learning_rate = " + str(lr) + "\n" 
            + "optimizing_function = " + str(opt_func) + "\n" 
            + "Loss = " + str(crit) + "\n" 
            + "batch_size = " + str(batch_size) + "\n"
            + "weight_decay = " + str(wd) + "\n")

    # create a file that contains the network architecture
    def writ_net(exp_folder,net):
        with open(exp_folder+'/network_architecture.txt', 'w') as f:
            f.write(str(net))

    # export the result of an expriment 
    def export_results(model, wd, batch_size, net, history, dataset, testloader_dataset, saving_location, sampling_factor,train_factor,num_epochs,lr,opt_func,crit): 
            exp_folder = Plot.new_folder(saving_location) # create folder
            # write hpyer parameters in a file
            Plot.write_param(exp_folder, wd, batch_size, sampling_factor,train_factor,num_epochs,lr,opt_func,crit)
            # write the network architecture in a file
            Plot.writ_net(exp_folder,net)
            # plot accuracies per epoch
            Plot.plot_accuracies(history, num_epochs, exp_folder+'/accuracy(e).png', exp_folder+'/accuracy(e).csv')
            # plot train and test losses per epoch
            Plot.plot_losses(history, num_epochs, exp_folder+'/losses.png', exp_folder+'/losses(e).csv')
            # plot accuracy per class (at the last epoch)
            Plot.class_accuracies(Plot, model, dataset, testloader_dataset, exp_folder+'/class_acc.csv')