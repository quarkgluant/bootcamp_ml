#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def vec_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrice of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector n * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg
    the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or theta.size == 0 or y.size == 0 or \
            (x.size != 0 and x.shape[1] != theta.shape[0]) or \
        x.shape[0] != y.shape[0]:
        return None
    # 1/m * transpose(x) * (x * theta - y)
    m = x.shape[0]
    return (1 / m) * np.dot(np.transpose(x), (np.dot(x, theta) - y))

if __name__ == '__main__':
    import numpy as np
    X = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    Z = np.array([3,0.5,-6])
    print(vec_gradient(X, Y, Z))
    # array([ -37.35714286, 183.14285714, -393.
    W = np.array([0,0,0])
    print(vec_gradient(X, Y, W))
    # array([ 0.85714286, 23.28571429, -26.42857143])
    print(vec_gradient(X, X.dot(Z), Z))
    # array([0., 0., 0.])