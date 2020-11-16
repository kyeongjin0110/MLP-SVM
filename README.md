# Multilayer Perceptrons (MLP) and Support Vector Machines (SVM)
#### The goal of this assignment is to solve the classification problem using Multilayer Perceptrons and Support Vector Machines. And compare the three methods (GMM, MLP and SVM) in various aspects.

## Main description

The data sets of two artificial classification problems will be used. 

Each data set is for a 2-class classification problem and consists of N training data and N test data,
where N=225 for problem 1 and N=220 for problem 2. The input data are within a range of [-1,1]^2,
and the target value is either 0 or 1, meaning class 1 or class 2, respectively. Four data files are
provided for each problem as follows (where ? is equal to 1 or 2):

    • p?_train_input.txt: training samples (Nx2 matrix)
    • p?_train_target.txt: target category (0 or 1) for each training sample (Nx1 vector)
    • p?_test_input.txt: test samples (Nx2 matrix)
    • p?_test_target.txt: target category for each test sample (Nx1 vector)

#### Multilayer perceptron (MLP)
Write your code for training and testing MLP classifiers. Try different structures (numbers of layers
and numbers of hidden neurons), types of activation functions, learning algorithms. Develop your
own research questions and conduct experiments to answer the questions. Report your results with
thorough discussion. Some example questions are (but not limited to):

    • How does the choice of the structure influence the performance?
    • How does the choice of the activation function influence the performance?
    • How does the choice of the learning algorithm influence the performance?
    • What is the optimal network structure? How is the ‘optimality’ defined?
    • Do you observe overfitting?

#### Support vector machine (SVM)
Write your code for training and testing SVM classifiers. Try different kernel functions and their
parameters, and penalty parameter (C) values. Develop your own research questions and conduct
experiments to answer the questions. Report your results with thorough discussion. Some example
questions are (but not limited to):

    • How does the choice of the kernel function and its parameter(s) influence the performance?
    • How does the choice of the penalty parameter influence the performance?
    • How many and which data are support vectors, and how do the support vectors change
    • Does overfitting occur?

Compare the three methods (GMM, MLP, and SVM) in various aspects. For example:

    • Which is better among GMM, MLP, and SVM? Why?
    • Is the superiority consistent between the two datasets?
    • Which one between the two datasets is easier to solve for which method?

## Requirements

- Ubuntu 16.04
- CUDA 10.1
- cuDNN 7.5
- Python 3.6
- sklearn 0.15.0.
- numpy 1.15.4
- matplotlib 2.1.0

## Testing

```bash

# GMM & visualization decision boundary
python3 GMM.py

# for testing with number of MLP parameters in all cases
python3 MLP_multi.py

# MLP best case's recall & fall-out graph
#  MLP best case's AUC
# MLP best case's decision boundary
python3 MLP.py

# for testing with number of SVM parameters in all cases
python3 SVM_multi.py

# SVM best case's recall & fall-out graph
# SVM best case's AUC
# SVM best case's decision boundary
python3 SVM.py
