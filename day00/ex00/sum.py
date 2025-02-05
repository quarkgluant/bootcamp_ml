#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def sum_(x, f):
    """Computes the sum of a non-empty numpy.ndarray onto wich a function is
    applied element-wise, using a for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    f: has to be a function, a function to apply element-wise to the
    vector.
    Returns:
    The sum as a float.
    None if x is an empty numpy.ndarray or if f is not a valid function.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0:
        return None
    summ = 0.0
    for element in x:
        try:
            summ += f(element)
        except:
            return None
    return summ


if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(sum_(X, lambda x: x))
    print(sum_(X, lambda x: x ** 2))

