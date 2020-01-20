#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def variance(x):
    """Computes the variance of a non-empty numpy.ndarray, using a for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    The variance as a float.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0:
        return None
    ft_mean = 0.0
    for element in x:
        try:
            ft_mean += element
        except:
            return None
    ft_mean = ft_mean / x.size
    ft_var = 0.0
    for element in x:
        try:
            ft_var += (element - ft_mean) ** 2
        except:
            return None
    return ft_var / x.size


if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(variance(X))
    # 134.57142857142858
    print(np.var(X))
    # 134.57142857142858
    print(variance(X / 2))
    # 33.642857142857146
    print(np.var(X / 2))
    # 33.642857142857146