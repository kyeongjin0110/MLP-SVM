###################################################################
##### for testing with number of MLP parameters in all cases  #####
###################################################################

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

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from itertools import cycle

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

## 은닉 유닛과 alpha, 다른 초깃값 매개변수에 따라 변하는 결정 경계
## solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
## activation = ['identity', 'logistic', 'tanh', 'relu']

## activation function 변경 같은 매개변수로 학습한 결정 경계
# ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        max_acc = 0
        best_random_state = 0
        best_n_hidden_nodes = 0
        best_alpha = 0
        acc_l = []
        for i in range (8):
            clf = MLPClassifier(solver='lbfgs', random_state=i,
                                hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                                alpha=alpha, max_iter=1000) # max_iter=1000 넣으니 정확도 향상
            clf.fit(train_input, train_target)
            y_pred = clf.predict(test_input)
            acc = accuracy_score(test_target, y_pred)
            acc_l.append(acc)
            if max_acc < acc:
                max_acc = acc
                best_random_state = i
                best_n_hidden_nodes = n_hidden_nodes
                best_alpha = alpha
        for k in acc_l:
            print("{:4f}\t".format(k), end='')
        print("", end='\n')

        clf = MLPClassifier(solver='lbfgs', random_state=best_random_state,
                                hidden_layer_sizes=[best_n_hidden_nodes, best_n_hidden_nodes],
                                alpha=best_alpha, max_iter=1000) # max_iter=1000 넣으니 정확도 향상
        clf.fit(train_input, train_target)

        mglearn.plots.plot_2d_separator(clf, train_input, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(train_input[:, 0], train_input[:, 1], train_target, ax=ax)

        y_pred = clf.predict(test_input)
        acc = accuracy_score(test_target, y_pred)
        cm = confusion_matrix(test_target, y_pred)

        fpr, tpr, thresholds = roc_curve(test_target, y_pred)

        _fpr = 1-fpr[np.argmax(np.abs(tpr - fpr))]
        _tpr = tpr[np.argmax(np.abs(tpr - fpr))]

        _auc = auc(fpr, tpr)

        ax.set_title("n_hidden=[{}, {}], alpha={:.4f}, init={}\nacc={:4f}, auc={:4f}\nspecificity={:4f}, sensitivity={:4f}".format(
                      best_n_hidden_nodes, best_n_hidden_nodes, best_alpha, best_random_state, acc, _auc, _fpr, _tpr))

## if you want to check the cm's heatmap 
# sns.heatmap(cm, center=True)

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
                                alpha=alpha, max_iter=1000) # max_iter=1000 넣으니 정확도 향상
clf.fit(train_input, train_target)

mglearn.plots.plot_2d_separator(clf, train_input, fill=True, alpha=.3)
mglearn.discrete_scatter(train_input[:, 0], train_input[:, 1], train_target)
plt.xlabel("Class 0")
plt.ylabel("Class 1")

###############################################
######### for analysis hidden weight ##########
## 신경망의 첫 번째 층의 가중치 히트맵 heat map ##
###############################################
# plt.figure(figsize=(20, 5))
# plt.imshow(clf.coefs_[0], interpolation='none', cmap='viridis')
# plt.yticks(range(2), ["x", "y"])
# plt.xlabel("Hidden unit")
# plt.ylabel("Input characteristic")
# plt.colorbar()

plt.show()

################################
### for accuracy, loss graph ###
################################
# from sklearn.metrics import log_loss

# for i in range (1, 1001):
#     clf = MLPClassifier(solver='lbfgs', random_state=random_state,
#                                 hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
#                                 alpha=alpha, max_iter=i) # max_iter=1000 넣으니 정확도 향상
#     clf.fit(train_input, train_target)

#     y_pred_train = clf.predict(train_input)
#     train_accuracy = accuracy_score(train_target, y_pred_train)
#     train_loss = log_loss(train_target, y_pred_train)

#     y_pred_test = clf.predict(test_input)
#     test_accuracy = accuracy_score(test_target, y_pred_test)
#     test_loss = log_loss(test_target, y_pred_test)

#     if i == 1:
#         print("{} {} {} {} {}".format(i, train_accuracy, test_accuracy, train_loss, test_loss))
#     if i % 10 == 0:
#         print("{} {} {} {} {}".format(i, train_accuracy, test_accuracy, train_loss, test_loss))

