import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="darkgrid", palette="deep")
sns.set_context("paper")

folder = '/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/Accuracies/'
DATA = pd.read_csv('/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/Accuracies/'+'BaseNet'+'.csv')
DATA = DATA.rename(columns = {'accuracies' :'BaseNet'})

file_names = ['ResNet-18','ResNet-34','DFL-VGG16']

for name in file_names:
    file = folder+name+'.csv'
    NewDF = pd.read_csv(file)
    NewDF = NewDF.rename(columns = {'accuracies' :name})
    DATA = DATA.join(NewDF[name])


DATA = DATA.drop(columns=["Unnamed: 0"])
DATA.drop([20,21,22,23,24,25,26,27,28,29], axis=0, inplace=True)
fig = sns.lineplot(data=DATA, dashes = False,markers= ["o","o","o","o"])
fig.set(xlabel='epochs', ylabel='accuracy')
#fig.set_linestyle("-")
plt.savefig(folder+'accuracies.eps')
plt.show()