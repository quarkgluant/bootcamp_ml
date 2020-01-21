#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def cost_elem_(theta, X, Y):
    """
    Description:
    Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost
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
    import numpy as np
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    print(cost_elem_(theta1, X1, Y1))
    # => array([[0.], [0.1], [0.4], [0.9], [1.6]])
    print(cost_(theta1, X1, Y1))
    # => 3.0
    X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    theta2 = np.array([[0.05], [1.], [1.], [1.]])
    Y2 = np.array([[19.], [42.], [67.], [93.]])
    print(cost_elem_(theta2, X2, Y2))
    # => array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])
    print(cost_(theta2, X2, Y2))
    # => 4.238750000000004