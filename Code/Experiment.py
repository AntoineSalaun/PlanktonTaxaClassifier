import torch.nn.functional as F
from time import time


from Net import *
from Model import *
from PlanktonLoader import *
from Plot import *


# The class exeperiments resembles a main. It contains all the usefull hyper-parameters, folder paths.
# It is initialized with parameters that are meant to be changed by the user.
# It does the following 1. get all the parameters 2. train the model 3. print/write the results
class Experiment:
    def __init__(self,
        sampling_factor = 1.0,
        train_factor = .7,
        num_epochs = 20,
        lr = 0.001,
        batch_size = 128,
        opt_func = torch.optim.SGD,
        wd = 0.2,
        crit = F.cross_entropy ,
        net = DFL_VGG16()):

        print('----------------New experiment---------------- ')
        print('sampling = ', sampling_factor, 'num_epochs = ', num_epochs, 'batch_size = ',  batch_size , 'lr = ', lr , 'opt_func = ', opt_func,'weight_decay', wd, 'crit = ', crit)
        print(net)

        data_folder = '../../Project_I/ZooScanSet' # to be changed if run in another environement
        normalize = ((0.5), (0.5)) # normalizes the images 
        unwanted_classes = ['seaweed','badfocus__Copepoda','artefact','badfocus__artefact','bubble','detritus','fiber__detritus','egg__other','multiple__other']

        transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(), transforms.Normalize(*normalize)]) #defines the pre-processing sequence
        dataset = PlanktonLoader(data_folder+'/taxa.csv', data_folder+"/imgs/", unwanted_classes ,transform) #load the data using the home-made dataloader
        trainloader_dataset, testloader_dataset = PlanktonLoader.build_loaders(dataset, sampling_factor, train_factor, batch_size, random_seed= 41, shuffle_dataset= True) #train-test split

        model = ImageClassificationBase(opt_func,wd,crit,net,lr,num_epochs) #instantiate a model
        history = model.fit(trainloader_dataset, testloader_dataset) # train the model
       
        print('starts ploting') 
        # calls a method that calls all the ploting methods
        Plot.export_results(model, wd, batch_size, net, history,dataset, testloader_dataset, '../../Project_I/Saving_Output/', sampling_factor,train_factor,num_epochs,lr,opt_func,crit )

        
