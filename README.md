# PlanktonTaxaClassifier

## Download the dataset

The dataset canbe downloaded from SEANOE's website : https://www.seanoe.org/data/00446/55741/ 
Then, move the data in the ZooScanSet folder

## Start a run
You can start a run by launching the following command 
```
cd Code
python3 Runner.py 1.0 0.7 20 1e-5 128 ADAM 0.1 ResNet18
```

## Training parameters 

The parameters should come after 
```
python3 Runner.py
```
in the following order
* sampling factor (What percentage of the data is used)
* train factor (How much of it is used for training, the rest is for validation)
* epochs
* learning rate
* Batch Size
* Optimizer (ADAM or SGD)
* Weight decay
* Network (BaseNet, ResNet18, ResNet34 or DFL-VGG16)

## Note

Relative paths are used but depending on the architecture of your file system, some paths might be broken. If it is the case, you should modify it in the beggining of Code/Experiment.py
