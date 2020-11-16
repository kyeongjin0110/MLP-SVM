################################################
### MLP best case's recall & fall-out graph  ###
###            MLP best case's AUC           ###
###    MLP best case's decision boundary     ###
################################################

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import datasets
from sklearn.externals.six.moves import xrange

from sklearn.neural_network import MLPClassifier
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import mglearn

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
from scipy import interp
from itertools import cycle

## Choose only one of the three flags.
recall_fallout = False
roc_curve_flag = False
mlp_show = True

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

if problem == 0:
    n_hidden_nodes = 100
    random_state = 6
    alpha = 1.0
else:
    n_hidden_nodes = 100
    random_state = 5
    alpha = 0.01

clf = MLPClassifier(solver='lbfgs', random_state=random_state,
                    hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                    alpha=alpha, max_iter=1000)

start = time.time()
clf.fit(train_input, train_target)
stop = time.time()
print(f"Training time: {stop - start}s")

y_pred_train = clf.predict(train_input)
train_accuracy = accuracy_score(train_target, y_pred_train)
print("train accuracy")
print(accuracy_score(train_target, y_pred_train))

y_pred_test = clf.predict(test_input)
test_accuracy = accuracy_score(test_target, y_pred_test)
print("test accuracy")
print(accuracy_score(test_target, y_pred_test))

cm = confusion_matrix(test_target, y_pred_test)
print("confusion_matrix")
print(cm)

print("classification_report")
print(classification_report(test_target, y_pred_test, target_names=['class 0', 'class 1']))

recall = cm[0][0] / (cm[0][0] + cm[0][1])
fallout = cm[1][0] / (cm[1][0] + cm[1][1])
print("recall = {:.5f}".format(recall))
print("fallout = {:.5f}".format(fallout))

fpr, tpr, thresholds = roc_curve(test_target, y_pred_test)

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

_y_pred_test = []
for pre in y_pred_test:
    if pre == 0.0:
        _y_pred_test.append([1, 0])
    else:
        _y_pred_test.append([0, 1])
_y_pred_test = np.array(_y_pred_test)

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
        fpr[i], tpr[i], _ = roc_curve(_test_target[:, i], _y_pred_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(_test_target.ravel(), _y_pred_test.ravel())
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

if mlp_show == True:
    if problem == 0:
        plt.text(-1.23, 1.15, 'Train accuracy: %.5f' % train_accuracy)
        plt.text(-1.23, 1.05, 'Test accuracy: %.5f' % test_accuracy)
        plt.text(-1.23, 0.95, 'Run time: %.5fs' % (stop - start))
    else:
        plt.text(-1.15, 1.48, 'Train accuracy: %.5f' % train_accuracy)
        plt.text(-1.15, 1.36, 'Test accuracy: %.5f' % test_accuracy)
        plt.text(-1.15, 1.24, 'Run time: %.5fs' % (stop - start))
    mglearn.plots.plot_2d_separator(clf, train_input, fill=True, alpha=.3)
    mglearn.discrete_scatter(train_input[:, 0], train_input[:, 1], train_target)
    plt.show()
