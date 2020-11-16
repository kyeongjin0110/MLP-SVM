###################################################################
##### for testing with number of SVM parameters in all cases  #####
###################################################################

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import datasets
from sklearn.externals.six.moves import xrange

from sklearn import svm
from sklearn.svm import SVC
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import mglearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

## Choose only one of the two flags.
gamma_change = True
C_change = False

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

problem = 0
# problem = 1

if problem == 0:
    train_input = np.loadtxt('./data/p1_train_input.txt')
    train_target = np.loadtxt('./data/p1_train_target.txt')
    test_input = np.loadtxt('./data/p1_test_input.txt')
    test_target = np.loadtxt('./data/p1_test_target.txt')
else:
    train_input = np.loadtxt('./data/p2_train_input.txt')
    train_target = np.loadtxt('./data/p2_train_target.txt')
    test_input = np.loadtxt('./data/p2_test_input.txt')
    test_target = np.loadtxt('./data/p2_test_target.txt')

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))

fig, ax = plt.subplots(1, 5, figsize=(35, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

if gamma_change == True:
    ## kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, default=’rbf’
    gamma_list = [1, 5, 10, 15, 20]
    for i, (axi, gamma) in enumerate (zip(ax, gamma_list)):
        clf = SVC(kernel='rbf', C=10, gamma=gamma, random_state=100)
        clf.fit(train_input, train_target)
        axi.scatter(train_input[:, 0], train_input[:, 1], c=train_target, s=50, cmap='autumn')
        # axi.scatter(test_input[:, 0], test_input[:, 1], c=test_target, s=50, cmap='autumn')
        plot_svc_decision_function(clf, axi)
        axi.scatter(clf.support_vectors_[:, 0],
                    clf.support_vectors_[:, 1],
                    s=300, lw=1, facecolors='none')

        y_pred = clf.predict(test_input)
        acc = accuracy_score(test_target, y_pred)
        print(accuracy_score(test_target, y_pred))
        # Model Precision: what percentage of positive tuples are labeled as such?
        precision = metrics.precision_score(test_target, y_pred)
        print("Precision:", metrics.precision_score(test_target, y_pred))
        # Model Recall: what percentage of positive tuples are labelled as such?
        recall = metrics.recall_score(test_target, y_pred)
        print("Recall:", metrics.recall_score(test_target, y_pred))

        axi.set_title("gamma = {:}\nacc = {:.4f}\nprecision = {}\nrecall = {}"
                        .format(gamma, acc, precision, recall), size=10)
if C_change == True:
    ## kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, default=’rbf’
    C_list_1 = [1.0, 1E1, 1E2, 1E3, 1E4]
    C_list_2 = [1E5, 1E6, 1E7, 1E8, 1E9]
    for i, (axi, C) in enumerate (zip(ax, C_list_1)):
        clf = SVC(kernel='rbf', C=C, gamma=10, random_state=100)
        clf.fit(train_input, train_target)
        axi.scatter(train_input[:, 0], train_input[:, 1], c=train_target, s=50, cmap='autumn')
        # axi.scatter(test_input[:, 0], test_input[:, 1], c=test_target, s=50, cmap='autumn')
        plot_svc_decision_function(clf, axi)
        axi.scatter(clf.support_vectors_[:, 0],
                    clf.support_vectors_[:, 1],
                    s=300, lw=1, facecolors='none')

        y_pred = clf.predict(test_input)
        acc = accuracy_score(test_target, y_pred)
        print(accuracy_score(test_target, y_pred))
        # Model Precision: what percentage of positive tuples are labeled as such?
        precision = metrics.precision_score(test_target, y_pred)
        print("Precision:", metrics.precision_score(test_target, y_pred))
        # Model Recall: what percentage of positive tuples are labelled as such?
        recall = metrics.recall_score(test_target, y_pred)
        print("Recall:", metrics.recall_score(test_target, y_pred))

        axi.set_title("C = {:}\nacc = {:.4f}\nprecision = {}\nrecall = {}"
                        .format(C, acc, precision, recall), size=10)
                    
plt.show()