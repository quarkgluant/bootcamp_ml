#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def mse(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.ndarray, using
    a for-loop. The two arrays must have the same dimensions.
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
    ft_temp = 0.0
    for x_item, y_item in np.dstack((y, y_hat))[0]:
        ft_temp += (x_item - y_item) ** 2
    return ft_temp / y.size


if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(mse(X, Y))
    # 4.285714285714286
    print(mse(X, X))
    # 0.0