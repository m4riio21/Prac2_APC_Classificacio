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
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv("../weatherAUS.csv")
dataset = dataset.replace('No',0)
dataset = dataset.replace('Yes',1)
dataset.drop('Date', inplace=True, axis=1)
dataset.drop('Location', inplace=True, axis=1)
dataset.drop('WindGustDir', inplace=True, axis=1)
dataset.drop('WindDir9am', inplace=True, axis=1)
dataset.drop('WindDir3pm', inplace=True, axis=1)



dataset = dataset.dropna(how='any',axis=0)

data = dataset.values

# Escalem les dades
X = data[:,:-1]
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)
y = data[:, -1]

param_svm = {'C': [0.1,1, 10, 100, 1000]}
param_perceptron = {'penalty': ['l2','l1'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                    'fit_intercept': [True, False], 'shuffle': [True, False]}
param_knn = { 'n_neighbors' : [5,7,9,11,13],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

param_logireg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

param_grid = [param_perceptron, param_knn, param_logireg, param_svm]

models = [Perceptron(), KNeighborsClassifier(), LogisticRegression(), svm.LinearSVC()]
nom_models = ["Perceptron", "KNN", "Logistic Regression", "Support Vector Machines"]

with open("../Results/ApartatB/hyperparameters/hyperparams_search.txt",'w') as f:
    for i,model in enumerate(models):
        print("Cerca hiperparàmetres, model "+nom_models[i])
        f.write("---- Cerca hiperparàmetres, model "+nom_models[i]+" ----\n\n")
        grid = GridSearchCV(model, param_grid[i], verbose=3, n_jobs=-1)
        grid.fit(x,y)
        print("Accuracy òptim amb valor: ",grid.best_score_, " amb paràmetres:",grid.best_params_)
        f.write("Accuracy òptim amb valor: "+str(grid.best_score_)+ " amb paràmetres: "+str(grid.best_params_)+"\n")
        print("")
        f.write("\n")