import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="darkgrid", palette="deep")
sns.set_context("paper")

folder = '/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/Optimizer/'
DATA = pd.read_csv('/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/Optimizer/'+'accuracy(e)_'+'ADAM_1e-3'+'.csv')
DATA = DATA.rename(columns = {'accuracies' :'ADAM_1e-3'})

file_names = ['ADAM_1e-4','ADAM_1e-5','SGD_1e-3','SGD_1e-4','SGD_1e-5']

for name in file_names:
    file = folder+'accuracy(e)_'+name+'.csv'
    NewDF = pd.read_csv(file)
    NewDF = NewDF.rename(columns = {'accuracies' :name})
    DATA = DATA.join(NewDF[name])


DATA = DATA.drop(columns=["Unnamed: 0"])

fig = sns.lineplot(data=DATA, dashes = False, markers=['o','o','o','o','o','o'])
fig.set(xlabel='epochs', ylabel='accuracy')
#fig.set_linestyle("+")
plt.savefig(folder+'accuracies.eps')
plt.show()