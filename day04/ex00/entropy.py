#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np
from math import log, e


def entropy(array):
    """
    Computes the Shannon Entropy of a non-empty numpy.ndarray
    :param numpy.ndarray array:
    :return float: shannon's entropy as a float or None if input is not a
    non-empty numpy.ndarray
    """
    if not isinstance(array, np.ndarray):
        print(f"Shannon entropy for {array} is {None}")
        return None

    n_labels = len(array)

    if n_labels <= 1:
        print(f"Shannon entropy for {array} is {0.0}")
        return 0

    value, counts = np.unique(array, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        print(f"Shannon entropy for {array} is {0.0}")
        return 0.0

    ent = 0.0

    # Compute entropy
    base = 2
    for i in probs:
        ent -= i * log(i, base)

    print(f"Shannon entropy for {array} is {ent}")
    return ent


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    entropy([])
    # Shannon entropy for [] is None
    entropy({1, 2})
    # Shannon entropy for {1, 2} is None
    entropy("bob")
    # Shannon entropy for bob is None
    entropy(np.array([0, 0, 0, 0, 0, 0]))
    # Shannon entropy for [0 0 0 0 0 0] is 0.0
    entropy(np.array([6]))
    # Shannon entropy for [6] is 0.0
    entropy(np.array(['a', 'a', 'b', 'b']))
    # Shannon entropy for ['a' 'a' 'b' 'b'] is 1.0
    entropy(np.array(['0', '0', '1', '0', 'bob', '1']))
    # Shannon entropy for ['0' '0' '1' '0' 'bob' '1'] is 1.4591479170272448
    entropy(np.array(['0', 'bob', '1']))
    # Shannon entropy for ['0' 'bob' '1'] is 1.5849625007
    entropy(np.array([0, 0, 1, 0, 2, 1]))
    # Shannon entropy for [0 0 1 0 2 1] is 1.459147917027
    entropy(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # Shannon entropy for [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    entropy(np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # Shannon entropy for [0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    entropy(np.array([0, 0, 1]))
    # Shannon entropy for [0 0 1] is 0.9182958340544896


