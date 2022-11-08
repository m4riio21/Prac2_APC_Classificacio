from warnings import filterwarnings

filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Eliminem atributs string que no ens serveixen i NaNs
dataset = pd.read_csv("../weatherAUS.csv")
dataset = dataset.replace('No', 0)
dataset = dataset.replace('Yes', 1)
dataset.drop('Date', inplace=True, axis=1)
dataset.drop('Location', inplace=True, axis=1)
dataset.drop('WindGustDir', inplace=True, axis=1)
dataset.drop('WindDir9am', inplace=True, axis=1)
dataset.drop('WindDir3pm', inplace=True, axis=1)
dataset = dataset.dropna(how='any', axis=0)

# Noms columnes
columns = []
for col in dataset.columns:
    columns.append(col)

data = dataset.values

# Agafem 10000 random rows (SVC)
number_of_rows = data.shape[0]
random_indices = np.random.choice(number_of_rows, size=10000, replace=False)
data_reduced = data[random_indices, :]

# Escalem les dades
X = data_reduced[:, :2]
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)
y = data_reduced[:, -1]

# No / Yes
n_classes = 2

fig, sub = plt.subplots(1, 2, figsize=(16, 6))
sub[0].scatter(x[:, 0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
sub[1].scatter(x[:, 1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')


particions = [0.5, 0.7, 0.8]

for part in particions:
    x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=part)

    # Creem el regresor logístic
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)

    # l'entrenem
    logireg.fit(x_t, y_t)

    print("Logistic regression with ", part * 100, "% of the data: ", logireg.score(x_v, y_v))

    # Creem el SVM
    svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)

    # l'entrenem
    svc.fit(x_t, y_t)
    probs = svc.predict_proba(x_v)
    print("SVM with ", part * 100, "% of the data: ", svc.score(x_v, y_v))

    # Creem el KNN
    knn = KNeighborsClassifier()
    knn.fit(x_t, y_t)
    print("KNN with", part * 100, "% of the data: ", knn.score(x_v, y_v))

    # Perceptron
    perceptron = Perceptron()
    perceptron.fit(x_t, y_t)
    print("Perceptron with ", part * 100, "% of the data: ", perceptron.score(x_v, y_v))

from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

models = [LogisticRegression(), svm.SVC(probability=True), KNeighborsClassifier(), CalibratedClassifierCV(Perceptron())]
nom_models = ["LogisticRegression", "SVM", "KNN", "Perceptron"]
for o, model in enumerate(models):
    x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=0.8)  # millor % test - validació trobat
    model.fit(x_t, y_t)
    probs = model.predict_proba(x_v)
    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}

    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])

        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
        plt.title(nom_models[o])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")
        plt.savefig("../Results/ApartatA/PR/PR-curve_"+nom_models[o]+".png")


    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    rnd_fpr, rnd_tpr, _ = roc_curve(y_v > 0, np.zeros(y_v.size))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.title(nom_models[o])
    plt.legend()
    plt.savefig("../Results/ApartatA/ROC/ROC-curve_" + nom_models[o] + ".png")

import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def show_C_effect(X,y,C=1.0, gamma=0.7, degree=3):

    # import some data to play with
#    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
#    X = iris.data[:, :2]
#    y = iris.target

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    #C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.savefig("../Results/ApartatA/C-effect/c-effect.png")

show_C_effect(x[:,:2], y, C=0.1)
