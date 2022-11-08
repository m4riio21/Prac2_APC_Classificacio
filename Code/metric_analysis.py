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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, average_precision_score, roc_curve, auc


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


models = [LinearSVC(random_state=0, tol=1e-5), Perceptron(), KNeighborsClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines - Linear", "Perceptron", "KNN", "Logistic Regression"]
nom_classes = ["No", "Yes"]
with open("../Results/ApartatB/metrics/scores.txt",'w') as f:
    for o, model in enumerate(models):
        print("---- ", nom_models[o], " ----")
        f.write("---- "+nom_models[o]+" ----\n\n")

        print("")
        x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=0.8)
        model.fit(x_t, y_t)
        prediccions = model.predict(x_v)
        print("Accuracy score: ", accuracy_score(y_v, prediccions))
        f.write("Accuracy score: "+ str(accuracy_score(y_v, prediccions))+"\n")
        print("F1 score: ", f1_score(y_v, prediccions, average='weighted'))
        f.write("F1 score: "+ str(f1_score(y_v, prediccions, average='weighted'))+"\n")
        print("Average precision score: ", average_precision_score(y_v, prediccions, average='weighted'))
        f.write("Average precision score: "+ str(average_precision_score(y_v, prediccions, average='weighted'))+"\n")
        print("Classification report: ", classification_report(y_v, prediccions, target_names=nom_classes))
        f.write("Classification report: "+ str(classification_report(y_v, prediccions, target_names=nom_classes))+"\n")
        f.write("\n")
