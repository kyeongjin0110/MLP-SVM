###############################################
#### GMM & visualization decision boundary ####
###############################################

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import time

def make_ellipses(gmm, ax):
    for n, color in enumerate('rg'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

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

X_total = np.concatenate((train_input, test_input), axis=0)
y_total = np.concatenate((train_target, test_target), axis=0)

target_names = ['0', '1']

## check number of components
# n_comps = np.arange(1, 100)
# clfs = [GMM(n, max_iter = 1000).fit(train_input[train_target==0]) for n in n_comps]
# bics = [clf.bic(train_input) for clf in clfs]
# aics = [clf.aic(train_input) for clf in clfs]

# plt.plot(n_comps, bics, label = 'BIC')
# plt.plot(n_comps, aics, label = 'AIC')
# plt.xlabel('n_components')
# plt.legend()
# plt.show()

## n_components = 60??
################################

if problem == 0:
    n_components = 10
else:
    n_components = 25

n_classes = len(np.unique(train_target))

# Try GMMs using different types of covariances.
gmm_class1_classifiers = dict((covar_type, GMM(n_components=n_components, covariance_type=covar_type, init_params='kmeans', random_state=0, tol=1e-3, max_iter=100))
                    for covar_type in ['spherical', 'diag', 'tied', 'full']) # full is default
gmm_class2_classifiers = dict((covar_type, GMM(n_components=n_components, covariance_type=covar_type, init_params='kmeans', random_state=0, tol=1e-3, max_iter=100))
                    for covar_type in ['spherical', 'diag', 'tied', 'full']) # full is default

n_gmm_class1_classifiers = len(gmm_class1_classifiers)
n_gmm_class2_classifiers = len(gmm_class2_classifiers)

plt.figure(figsize=(3 * n_gmm_class1_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)

## for gmm_class1_classifiers
for index, _classifier in enumerate (zip(gmm_class1_classifiers.items(), gmm_class2_classifiers.items())):

    # c1_classifier.means_ = np.array([train_input[train_target == i].mean(axis=0)
    #                             for i in xrange(n_classes)])
    # print(len(_classifier))
    c1_name = _classifier[0][0]
    c1_classifier = _classifier[0][1]
    c2_name = _classifier[1][0]
    c2_classifier = _classifier[1][1]

    c1_classifier.fit(train_input[train_target==0])
    c2_classifier.fit(train_input[train_target==1])

    # ## Akaike information criterion for the current model on the input X.
    # c1_aic = c1_classifier.aic(train_input)
    # print(c1_aic)
    # c1_aic = c1_classifier.aic(train_input[train_target==0])
    # print(c1_aic)
    # ## Bayesian information criterion for the current model on the input X.
    # c1_bic = c1_classifier.bic(train_input)
    # print(c1_bic)
    # c1_bic = c1_classifier.bic(train_input[train_target==0])
    # print(c1_bic)

    # Compute the weighted log probabilities for each sample.
    train_output_class1 = c1_classifier.score_samples(train_input)
    train_output_class2 = c2_classifier.score_samples(train_input)

    train_output = train_output_class1 < train_output_class2
    # print(train_output)
    train_accuracy = sum(train_output==train_target) / train_output.size * 100
    # print(train_accuracy)

    test_output_class1 = c1_classifier.score_samples(test_input)
    test_output_class2 = c2_classifier.score_samples(test_input)
    test_output = test_output_class1 < test_output_class2
    test_accuracy = sum(test_output==test_target) / test_output.size * 100
    # print(test_accuracy)

    # draw for c1
    classifier = c1_classifier
    n_classifiers = n_gmm_class1_classifiers
    name = c1_name

    ## draw for c2
    # classifier = c2_classifier
    # n_classifiers = n_gmm_class2_classifiers
    # name = c2_name

    h = plt.subplot(2, n_classifiers / 2, index + 1)
    # make_ellipses(classifier, h)

    for n, color in enumerate('rg'):
        data = X_total[y_total == n]
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate('rg'): # CMYK # rg
        data = test_input[test_target == n]
        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
            transform=h.transAxes)

    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
            transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

    ## visualization decision boundary ##
    hh = .02  # step size in the mesh
    x_min, x_max = train_input[:, 0].min() - 1, train_input[:, 0].max() + 1
    y_min, y_max = train_input[:, 1].min() - 1, train_input[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hh),
                        np.arange(y_min, y_max, hh))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max]
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    plt.axis('off')
    ########################################

plt.legend(loc='lower right', prop=dict(size=12))
plt.show()