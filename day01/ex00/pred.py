#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def predict_(theta, X):
    """
    Description:
    Prediction of output using the hypothesis function (linear model).
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension (number of
    features + 1, 1).
    X: has to be a numpy.ndarray, a matrix of dimension (number of
    training examples, number of features).
    Returns:
    pred: numpy.ndarray, a vector of dimension (number of the training
    examples,1).
    None if X does not match the dimension of theta.
    Raises:
    This function should not raise any Exception.
    """
    if X.size == 0 or theta.size == 0 or \
            (X.size != 0 and X.shape[1] + 1 != theta.shape[0]):
        return None
    # on commence par rajouter une colonne x0 qui vaut 1
    X = np.insert(X, 0, [1.], axis=1)
    return np.dot(X, theta)
    
if __name__ == '__main__':
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    print(predict_(theta1, X1))
    # => array([[2], [6], [10], [14.], [18.]])
    X2 = np.array([[1], [2], [3], [5], [8]])
    theta2 = np.array([[2.]])
    print(predict_(theta2, X2))
    # Incompatible dimension match between X and theta.
    # =>  None
    X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    theta3 = np.array([[0.05], [1.], [1.], [1.]])
    print(predict_(theta3, X3))
    # => array([[22.25], [44.45], [66.65], [88.85]])
    # """réponses aux questions du sujet:
    # Q: What is a hypothesis and what is it goal ?
    # A: An hypothesis is an explanation about the relationship between data populations that is interpreted
    # probabilistically and a candidate model that approximates a target function for mapping inputs to outputs
    # Q: Considering a training set with 4242 examples and 3 features. How many components are
    # there in the vector ?
    # A: the theta vector has shape (4, 1)
    # Q: Considering the vector has a shape (6,1) and the output has the shape (7,1). What is the
    # shape of the training set X ?
    # A: X[x ,y] x theta[6, 1] = output[7, 1]
    #  so, x = 7 and y = 5 (+ 1), so X.shape = (7, 5)
    # """