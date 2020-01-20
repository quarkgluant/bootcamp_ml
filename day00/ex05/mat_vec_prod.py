#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def mat_vec_prod(x, y):
    """Computes the product of two non-empty numpy.ndarray, using a
    for-loop. The two arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension n * 1.
    Returns:
    The product of the matrix and the vector as a vector of dimension m *
    1.
    None if x or y are empty numpy.ndarray.
    None if x and y does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
"""
    if x.size == 0 or y.size == 0 or (x.size != 0 and x.shape[1] != y.shape[0]) or y.shape[1] != 1:
        return None
    ft_prod = np.zeros((x.shape[0], 1), dtype=int)
    for index, vector in enumerate(x):
        ft_temp = 0
        for x_item, y_item in np.dstack((vector, y.reshape(vector.shape)))[0]:
            ft_temp += x_item * y_item
        ft_prod[index] = int(ft_temp)
    return ft_prod


if __name__ == '__main__':
    W = np.array([
        [-8, 8, -6, 14, 14, -9, -4],
        [2, -11, -2, -11, 14, -2, 14],
        [-13, -2, -5, 3, -8, -4, 13],
        [2, 13, -14, -15, -14, -15, 13],
        [2, -1, 12, 3, -7, -3, -6]])
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7, 1))
    print(mat_vec_prod(W, X))
    # mat_vec_prod(W, X)
    # array([[497],
    #        [-356],
    #        [-345],
    #        [-270],
    #        [-69]])
    print(W.dot(X))
    # array([[497],
    #        [-356],
    #        [-345],
    #        [-270],
    #        [-69]])
    print(mat_vec_prod(W, Y))
    # array([[452],
    #        [-285],
    #        [-333],
    #        [-182],
    #        [-133]])
    print(W.dot(Y))
    # array([[452],
    #        [-285],
    #        [-333],
    #        [-182],
    #        [-133]])
