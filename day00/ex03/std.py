#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def std(x):
    """Computes the standard deviation of a non-empty numpy.ndarray, using a
    for-loop.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    The standard deviation as a float.
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
    return (ft_var / x.size) ** 0.5


if __name__ == '__main__':
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(std(X))
    # 11.600492600378166
    print(np.std(X))
    # 11.600492600378166
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(std(Y))
    # 11.410700312980492
    print(np.std(Y))
    # 11.410700312980492