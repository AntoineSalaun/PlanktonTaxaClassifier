import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="darkgrid", palette="deep")
sns.set_context("paper")

folder = '/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/weight_decay/'
DATA = pd.read_csv('/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/weight_decay/'+'accuracy(e)_'+'wd=0,0'+'.csv')
DATA = DATA.rename(columns = {'accuracies' :'wd=0,0'})

file_names = ['wd=0,1','wd=0,2','wd=0,3']

for name in file_names:
    file = folder+'accuracy(e)_'+name+'.csv'
    NewDF = pd.read_csv(file)
    NewDF = NewDF.rename(columns = {'accuracies' :name})
    DATA = DATA.join(NewDF[name])

DATA = DATA.drop(columns=["Unnamed: 0"])

fig = sns.lineplot(data=DATA, dashes = False,markers=['o','o','o','o'])
fig.set(xlabel='epochs', ylabel='accuracy')
#fig.set_linestyle("+")
plt.savefig(folder+'accuracies.eps')
plt.show()