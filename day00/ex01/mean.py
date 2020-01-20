#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def mean(x):
    """Computes the mean of a non-empty numpy.ndarray, using a for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    The mean as a float.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0:
        return None
    summ = 0.0
    for element in x:
        try:
            summ += element
        except:
            return None
    return summ / x.size



if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(mean(X))
    # 1.0
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(mean(X ** 2))
    # 135.57142857142858