#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def fit_(theta, X, Y, alpha, n_cycle):
    """
    Description:
    Performs a fit of Y(output) with respect to X.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension (number of
    features + 1, 1).
    X: has to be a numpy.ndarray, a matrix of dimension (number of
    training examples, number of features).
    Y: has to be a numpy.ndarray, a vector of dimension (number of
    training examples, 1).
    alpha: positive float
    n_cycle: positive integer
    Returns:
    new_theta: numpy.ndarray, a vector of dimension (number of the
    features +1,1).
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """
    if X.size == 0 or theta.size == 0 or Y.size == 0 or \
            (X.size != 0 and X.shape[1] + 1 != theta.shape[0]) or \
        X.shape[0] != Y.shape[0]:
        return None
    # on commence par rajouter une colonne x0 qui vaut 1
    X_1 = np.insert(X, 0, [1.], axis=1)
    # 1/m * transpose(X) * (X * theta - Y)
    m = X.shape[0]
    for _ in range(n_cycle):
        # theta = theta - (alpha / m) * np.dot(np.transpose(X_1), (np.dot(X_1, theta) - Y))
        theta = theta - (alpha / m) * np.dot(np.transpose(X_1), predict_(theta, X) - Y)
    return theta


def cost_elem_(theta, X, Y):
    """
    Description:
    Calculates all the elements 0.5/M*(y_pred - y)^2 of the cost
    function.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension (number of
    features + 1, 1).
    X: has to be a numpy.ndarray, a matrix of dimension (number of
    training examples, number of features).
    Returns:
    J_elem: numpy.ndarray, a vector of dimension (number of the training
    examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    Raises:
    This function should not raise any Exception.
    """
    if X.size == 0 or theta.size == 0 or Y.size == 0 or \
            (X.size != 0 and X.shape[1] + 1 != theta.shape[0]) or \
        X.shape[0] != Y.shape[0]:
        return None
    # on commence par rajouter une colonne x0 qui vaut 1
    X_1 = np.insert(X, 0, [1.], axis=1)
    Y_hat = np.dot(X_1, theta)
    m = X.shape[0]
    return 0.5 / m * (Y_hat - Y) ** 2


def cost_(theta, X, Y):
    """
    Description:
    Calculates the value of cost function.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension (number of
    features + 1, 1).
    X: has to be a numpy.ndarray, a vector of dimension (number of
    training examples, number of features).
    Returns:
    J_value : has to be a float.
    None if X does not match the dimension of theta.
    Raises:
    This function should not raise any Exception.
    """
    return cost_elem_(theta, X, Y).sum()


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
    X_1 = np.insert(X, 0, [1.], axis=1)
    return np.dot(X_1, theta)


if __name__ == '__main__':
    import numpy as np
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
    theta1 = np.array([[1.], [1.]])
    theta1 = fit_(theta1, X1, Y1, alpha=0.01, n_cycle=2000)
    print(theta1)
    # array([[2.0023..], [3.9991..]])
    print(predict_(theta1, X1))
    # array([2.0023..], [6.002..], [10.0007..], [13.99988..], [17.9990..])

    X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta2 = np.array([[42.], [1.], [1.], [1.]])
    theta2 = fit_(theta2, X2, Y2, alpha=0.0005, n_cycle=42000)
    print(theta2)
    # array([[41.99..], [0.97..], [0.77..], [-1.20..]])
    print(predict_(theta2, X2))
    # array([[19.5937..], [-2.8021..], [-25.1999..], [-47.5978..]])