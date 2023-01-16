import torch 
from torchvision import transforms
import torch.nn.functional as F

from Net import *
from Model import *
from PlanktonLoader import *
from Plot import *
from Experiment import *
import sys


############################################## 
# Runner.py is used to run experiments. 
# it recieves the argument through the terminal and instatiate an class Experiment to make it run
# It can also be used by entering the arguments directly in the experiment method (and serve as a kind of init file)
# However, this was used like this because when running several experiments on the cluster,
# It will use the current version of it when the job actually starts
# If the Runner file has been modified while the job was pending, it will use the current configuration
# Using arguments in the sbatch command lets us start several jobs with different parameters at the same time.


# When calling Runner.py give the parameters in the right order :
# sampling_factor, train_factor, num_epochs, learning_rate, batch_size, 
# optimzer ('ADAM' or 'SGD'), weight_decay, Network ('BaseNet', 'ResNet18', 'ResNet34' or 'DFL')

# Receive the string argument and instantiate the right corresponding class
if sys.argv[6] == 'SGD':
    opti = torch.optim.SGD
if sys.argv[6]  == 'ADAM':
    opti = torch.optim.Adam

if sys.argv[8] == 'DFL':
    network = DFL_VGG16()
if sys.argv[8] == 'ResNet18':
    network = ResNet18()
if sys.argv[8] == 'ResNet34':
    network = ResNet34()
if sys.argv[8] == 'BaseNet':
    network = BaseNet()

# runs an experiment (and converts the parameters to the right type)
exp = Experiment(
sampling_factor = float(sys.argv[1]),
train_factor = float(sys.argv[2]),
num_epochs = int(sys.argv[3]),
lr = float(sys.argv[4]),
batch_size = int(sys.argv[5]),
opt_func = opti,
wd = float(sys.argv[7]),
crit = F.cross_entropy,
net = network)