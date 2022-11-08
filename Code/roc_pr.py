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
nom_models = ["LinearSVM", "Perceptron", "KNN", "LogisticRegression"]
nom_classes = ["No", "Yes"]

for i, model in enumerate(models):
    model = CalibratedClassifierCV(models[i])
    x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=0.8)
    model.fit(x_t, y_t)
    probs = model.predict_proba(x_v)
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for j in range(2):
        precision[j], recall[j], _ = precision_recall_curve(y_v == j, probs[:, j])
        average_precision[j] = average_precision_score(y_v == j, probs[:, j])

        plt.plot(recall[j], precision[j],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(j, average_precision[j]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(nom_models[i])
        plt.legend(loc="upper right")
    plt.savefig("../Results/ApartatB/PR/corba-pr_" + str(nom_models[i]) + ".png")

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for j in range(2):
        fpr[j], tpr[j], _ = roc_curve(y_v == j, probs[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    rnd_fpr, rnd_tpr, _ = roc_curve(y_v > 0, np.zeros(y_v.size))
    for j in range(2):
        plt.plot(fpr[j], tpr[j], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(j, roc_auc[j]))
    plt.title(nom_models[i])
    plt.legend()
    plt.savefig("../Results/ApartatB/ROC/corba-roc_" + str(nom_models[i]) + ".png")