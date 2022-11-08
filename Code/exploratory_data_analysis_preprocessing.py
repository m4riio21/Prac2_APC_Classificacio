import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, average_precision_score, roc_curve, auc

#Importem la BD
dataset = pd.read_csv("../weatherAUS.csv")

#Substitució dels atributs No/Yes per 0/1 per poder generar gràfiques
dataset = pd.read_csv("../weatherAUS.csv")
dataset = dataset.replace('No',0)
dataset = dataset.replace('Yes',1)

#Eliminem atributs de tipus string per poder fer el escalat de les dades
dataset.drop('Date', inplace=True, axis=1)
dataset.drop('Location', inplace=True, axis=1)
dataset.drop('WindGustDir', inplace=True, axis=1)
dataset.drop('WindDir9am', inplace=True, axis=1)
dataset.drop('WindDir3pm', inplace=True, axis=1)

#Drop NaNs
dataset = dataset.dropna(how='any',axis=0)

data = dataset.values

# Escalem les dades --> preprocessing
X = data[:,:-1]
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)
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
for i in range(x.shape[1]):
    sns.histplot(data=dataset, x=x[:, i])
    plt.xlabel(columns[i])
    plt.savefig("../Results/ApartatB/histogramas/atribut" + str(i) + ".png")
    plt.clf()

#Dispersion
plt.figure()
sns.set()
titles = dataset.columns.values
for i in range(x.shape[1]):
    sns.scatterplot(data=dataset, x=titles[i], y=titles[-1], hue="RainTomorrow")
    plt.savefig("../results/ApartatB/dispersion/atribut" + str(i) + ".png")
    plt.clf()