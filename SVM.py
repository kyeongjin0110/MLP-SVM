################################################
### SVM best case's recall & fall-out graph  ###
###            SVM best case's AUC           ###
###    SVM best case's decision boundary     ###
################################################

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
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
from scipy import interp
from itertools import cycle

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

## Choose only one of the three flags.
recall_fallout = False
roc_curve_flag = False
svm_show = True

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

clf = SVC(kernel='rbf', C=10, gamma=10, random_state=100)
clf.fit(train_input, train_target)

y_pred = clf.predict(test_input)

cm = confusion_matrix(test_target, y_pred)
print("confusion_matrix")
print(cm)

print("classification_report")
print(classification_report(test_target, y_pred, target_names=['class 0', 'class 1']))

recall = cm[0][0] / (cm[0][0] + cm[0][1])
fallout = cm[1][0] / (cm[1][0] + cm[1][1])
print("recall = {:.5f}".format(recall))
print("fallout = {:.5f}".format(fallout))

fpr, tpr, thresholds = roc_curve(test_target, y_pred)
_fpr = 1-fpr[np.argmax(np.abs(tpr - fpr))]
_tpr = tpr[np.argmax(np.abs(tpr - fpr))]
print("Specificity = {:.5f}".format(_fpr))
print("Sensitivity = {:.5f}".format(_tpr))

_auc = auc(fpr, tpr)
print("AUC = {:.5f}".format(_auc))

########################################
############ Draw ROC curve ############
########################################
lw = 2
n_classes = 2

_test_target = []
for tar in test_target:
    if tar == 0.0:
        _test_target.append([1, 0])
    else:
        _test_target.append([0, 1])
_test_target = np.array(_test_target)

_y_pred = []
for pre in y_pred:
    if pre == 0.0:
        _y_pred.append([1, 0])
    else:
        _y_pred.append([0, 1])
_y_pred = np.array(_y_pred)

if recall_fallout == True:
    plt.plot(fpr, tpr, 'o-', label="Logistic Regression")
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")
    plt.plot([fallout], [recall], 'ro', ms=10)
    plt.xlabel('Fall-Out')
    plt.ylabel('Recall')
    plt.show()

if roc_curve_flag == True:
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(_test_target[:, i], _y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(_test_target.ravel(), _y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    ## Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.5f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.5f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.5f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

if svm_show == True:
    plt.scatter(train_input[:, 0], train_input[:, 1], c=train_target, s=50, cmap='autumn')
    plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
    plt.show()