import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../weatherAUS.csv")
dataset = dataset.dropna(how='any',axis=0)

data = dataset.values

x = data[:,:-1]
y = data[:, -1]

#Mapa de calor (correlation)
correlacio = dataset.corr()
plt.figure()
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.savefig("../Results/ApartatB/correlation/correlation.png")
plt.clf()

#Histogramas
#Target
sns.histplot(data=dataset, x=y)
plt.xlabel('RainTomorrow')
plt.savefig("../Results/ApartatB/histogramas/atributTarget.png")
plt.clf()

columns = []
for col in dataset.columns:
    columns.append(col)
#X
for i in range(22):
    sns.histplot(data=dataset, x=x[:, i])
    plt.xlabel(columns[i])
    plt.savefig("../Results/ApartatB/histogramas/atribut" + str(i) + ".png")
    plt.clf()

#Dispersion
plt.figure()
sns.set()
titles = dataset.columns.values
for i in range(22):
    sns.scatterplot(data=dataset, x=titles[i], y=titles[-1], hue="RainTomorrow")
    plt.savefig("../results/ApartatB/dispersion/atribut" + str(i) + ".png")
    plt.clf()