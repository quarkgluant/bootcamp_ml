#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def gini(array):
    """
    Computes the gini impurity of a non-empty numpy.ndarray
    :param numpy.ndarray array:
    :return float: gini_impurity as a float or None if input is not a
    non-empty numpy.ndarray
    """
    if not isinstance(array, np.ndarray):
        print(f"Gini impurity for {array} is {None}")
        return None

    n_labels = len(array)

    if n_labels <= 1:
        print(f"Gini impurity for {array} is {0.0}")
        return 0.0

    value, counts = np.unique(array, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    gini = 1 - np.sum(probs**2)
    print(f"Gini impurity for {array} is {gini}")
    return gini


if __name__ == '__main__':
    # Output examples:
    gini([])
    # Gini impurity for [] is None  
    gini({1, 2})
    # Gini impurity for {1, 2} is None   
    gini("bob")
    # Gini impurity for bob is None  
    gini(np.array([0, 0, 0, 0, 0, 0]))
    # Gini impurity for [0 0 0 0 0 0] is 0.0    
    gini(np.array([6]))
    # Gini impurity for [6] is 0.0 
    gini(np.array(['a', 'a', 'b', 'b']))
    # Gini impurity for ['a' 'a' 'b' 'b'] is 0.5                         
    gini(np.array(['0', '0', '1', '0', 'bob', '1']))
    # Gini impurity for ['0' '0' '1' '0' 'bob' '1'] is 0.6111111111111112
    gini(np.array(['0', 'bob', '1']))
    # Gini impurity for ['0' 'bob' '1'] is 0.6666666666666667
    gini(np.array([0, 0, 1, 0, 2, 1]))
    # Gini impurity for [0 0 1 0 2 1] is 0.6111111111111112
    gini(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # Gini impurity for [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] is 0.0
    gini(np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # Gini impurity for [0. 1. 1. 1. 1. 1. 1. 1. 1. 1.] is 0.18
    gini(np.array([0, 0, 1]))
    # Gini impurity for [0 0 1] is 0.4444444444444445
