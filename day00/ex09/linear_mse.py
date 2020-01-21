#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def linear_mse(x, y, theta):
    """Computes the mean squared error of three non-empty numpy.ndarray,
    using a for-loop. The three arrays must have compatible dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    x: has to be an numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be an numpy.ndarray, a vector of dimension n * 1.
    Returns:
    The mean squared error as a float.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or theta.size == 0 or\
            (x.size != 0 and x.shape[1] != theta.shape[0]) or \
            x.shape[0] != y.shape[0]:
        return None
    # h = x.dot(theta)
    h = np.zeros((x.shape[1], theta.shape[0]))
    sigma = []
    ft_prod = np.zeros((x.shape[0], 1), dtype=int)
    for index, vector in enumerate(x):
        ft_temp = 0
        for x_item, y_item in np.dstack((vector, y.reshape(vector.shape)))[0]:
            ft_temp += x_item * y_item
        ft_prod[index] = int(ft_temp)


    delta = np.asarray(sigma)
    sigma = 0.0
    for delta_item, y_item in np.dstack((delta, y))[0]:
        sigma += (delta_item - y_item) ** 2
    return sigma / x.shape[0]


if __name__ == '__main__':
    X = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    Z = np.array([3, 0.5, -6])
    print(linear_mse(X, Y, Z))
    # 2641.0
    W = np.array([0, 0, 0])
    print(linear_mse(X, Y, W))
    # 130.71428571