#import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#exec('%matplotlib inline')

#dataset = pd.read_csv('data.csv')
#sns.pairplot(dataset, hue='Class')


import seaborn as sns
import matplotlib as plt

sns.set()
dataset = pd.read_csv('data.csv')
corr = dataset.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#print(dataset)