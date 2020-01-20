#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def vec_mse(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.ndarray,
    without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.ndarray.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if y.size == 0 or y_hat.size == 0 or (y.size != 0 and y_hat.shape[0] != y.shape[0]):
        return None
    return (np.transpose(y - y_hat) * (y - y_hat)).sum() / y.size


if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(vec_mse(X, Y))
    # 4.285714285714286
    print(vec_mse(X, X))
    # 0.0