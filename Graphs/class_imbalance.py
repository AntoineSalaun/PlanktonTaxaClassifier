import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="darkgrid", palette="deep")
sns.set_context("paper")

folder = '/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/Class_imbalance/'
DATA = pd.read_csv('/Users/nounou/Desktop/EPFL/M5/Project_I/Graphs/Graph_data/Class_imbalance/'+'class_imbalance'+'.csv')
#DATA = DATA.rename(columns = {'accuracies' :'BaseNet'})

file_names = ['class_pop']
unwanted_classes = ['seaweed','badfocus__Copepoda','artefact','badfocus__artefact','bubble','detritus','fiber__detritus','egg__other','multiple__other']

for name in file_names:
    file = folder+name+'.csv'
    NewDF = pd.read_csv(file)
    NewDF = NewDF.rename(columns = {'Unnamed: 0' :'taxon'})
    #print(NewDF)
    NewDF = NewDF[~NewDF['taxon'].isin(unwanted_classes)]
    NewDF= NewDF.set_index('taxon')
    #print(NewDF)

DATA = DATA.rename(columns = {'Class' :'taxon'})
DATA = DATA[~DATA['taxon'].isin(unwanted_classes)]
DATA = DATA.drop(columns=["Unnamed: 0"])
DATA= DATA.set_index('taxon')
DATA = DATA.join(NewDF)
DATA.columns =['pred','acc','pop']
DATA= DATA.sort_values(by='pred',ascending=True)

print(DATA)
palette = iter(sns.husl_palette(5))

plt.figure(figsize=(6,4))
fig = sns.scatterplot(data=DATA, x='pop', y='acc')
sns.lineplot(data=DATA, x='pop', y= 0.645, color=next(palette), linestyle="dashed")

plt.xscale('log')
fig.set_ylim(0, 1)
fig.set(xlabel='class population', ylabel='top-1 accuracy')
plt.legend(labels=['class accuracies','global accuracy = 0.645'])
#fig.set_linestyle("-")
plt.savefig(folder+'class_acc.eps')
plt.show()