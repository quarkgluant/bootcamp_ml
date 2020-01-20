#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def dot(x, y):
    """Computes the dot product of two non-empty numpy.ndarray, using a
    for-loop. The two arrays must have the same dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector.
    y: has to be an numpy.ndarray, a vector.
    Returns:
    The dot product of the two vectors as a float.
    None if x or y are empty numpy.ndarray.
    None if x and y does not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or (x.size != 0 and x.size != y.size):
        return None
    ft_dot = 0.0
    for x_item, y_item in np.dstack((x, y))[0]:
        ft_dot += x_item * y_item
    return ft_dot



if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(dot(X, Y))
    print(np.dot(X, Y))
    print(dot(X, X))
    # 949.0
    print(np.dot(X, X))
    # 949
    print(dot(Y, Y))
    # 915.0
    print(np.dot(Y, Y))
    # 915