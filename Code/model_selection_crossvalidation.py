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

tests = [0.5, 0.8, 0.7]
from sklearn.svm import LinearSVC
with open("../Results/ApartatB/models/models.txt",'w') as f:
    for t in tests:
        x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=t)
        f.write("---- "+str(t*100)+"% Test - "+str((1-t)*100)+"% Validation ----\n\n")
        # Logistic regressor
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
        logireg.fit(x_t, y_t)
        f.write("Logistic Regressor with  "+ str(t*100)+ "% of the data: "+ str(logireg.score(x_v, y_v))+"\n")
        print("Logistic Regressor with  ", t*100, "% of the data: ", logireg.score(x_v, y_v))


        # Creem el KNN
        knn = KNeighborsClassifier()
        knn.fit(x_t, y_t)
        f.write("KNN with "+ str(t*100)+"% of the data: "+ str(knn.score(x_v, y_v))+"\n")
        print("KNN with", t*100, "% of the data: ", knn.score(x_v, y_v))



        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(x_t, y_t)
        f.write("SVM with "+ str(t*100)+ "% of the data: "+ str(clf.score(x_v, y_v))+"\n")
        print("SVM with ", t*100, "% of the data: ", clf.score(x_v, y_v))


        # Perceptron
        perceptron = Perceptron()
        perceptron.fit(x_t, y_t)
        f.write("Perceptron with "+ str(t*100)+"% of the data: "+ str(perceptron.score(x_v, y_v))+"\n")
        print("Perceptron with ", t*100, "% of the data: ", perceptron.score(x_v, y_v))


        print("")
        f.write("\n")

    f.write("---- K-fold scores ----\n\n")
    for k in range(2, 7):
        scores = cross_val_score(logireg, x, y, cv=k)
        print("Logistic Regressor amb k-fold = ", k, " : ", scores.mean())
        f.write("Logistic Regressor amb k-fold = "+ str(k)+ " : "+ str(scores.mean())+"\n")
    f.write("\n")
    for k in range(2, 7):
        scores = cross_val_score(knn, x, y, cv=k)
        print("KNN amb k-fold = ", k, " : ", scores.mean())
        f.write("KNN amb k-fold = "+ str(k)+ " : "+ str(scores.mean())+ "\n")
    f.write("\n")
    for k in range(2, 7):
        scores = cross_val_score(clf, x, y, cv=k)
        print("SVM amb k-fold = ", k, " : ", scores.mean())
        f.write("SVM amb k-fold = "+ str(k)+ " : "+ str(scores.mean())+ "\n")
    f.write("\n")
    for k in range(2, 7):
        scores = cross_val_score(perceptron, x, y, cv=k)
        f.write("Perceptron amb k-fold = "+ str(k)+ " : "+ str(scores.mean())+ "\n")
        print("Perceptron amb k-fold = ", k, " : ", scores.mean())





