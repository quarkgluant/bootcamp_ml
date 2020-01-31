#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np
from math import log, e

def information_gain(array_source, array_children_list, criterion='gini'):
    """
    Computes the information gain between the first and second array using
    the criterion ('gini' or 'entropy')
    :param numpy.ndarray array_source:
    :param list array_children_list: list of numpy.ndarray
    :param str criterion: Should be in ['gini', 'entropy']
    :return float: Shannon entropy as a float or None if input is not a
    non-empty numpy.ndarray or None if invalid input
    """

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
    # Output examples:
    information_gain([], np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), criterion='gini')
    information_gain([], np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), criterion='entropy')
    # Information gain between[] and [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] is None with criterion 'gini' and None with
    # criterion 'entropy'

    information_gain(np.array(['a', 'a', 'b', 'b']), {1, 2}, criterion='gini')
    information_gain(np.array(['a', 'a', 'b', 'b']), {1, 2}, criterion='entropy')
    # Information gain between['a' 'a' 'b' 'b'] and {1, 2} is None with criterion 'gini' and None with criterion
    # 'entropy'

    information_gain(np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), criterion='gini')
    information_gain(np.array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), criterion='entropy')
    # Information gain between[0. 1. 1. 1. 1. 1. 1. 1. 1. 1.] and [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] is 0.18 with
    # criterion 'gini' and 0.4689955935892812 with criterion 'entropy'

    information_gain(np.array(['0', '0', '1', '0', 'bob', '1']), np.array(['0', 'bob', '1']), criterion='gini')
    information_gain(np.array(['0', '0', '1', '0', 'bob', '1']), np.array(['0', 'bob', '1']), criterion='entropy')
    # Information gain between['0' '0' '1' '0' 'bob' '1'] and [array(['0', 'bob', '1'], dtype='

    information_gain(np.array(['0', '0', '1', '0', 'bob', '1']), np.array([0, 0, 1, 0, 2, 1]), criterion='gini')
    information_gain(np.array(['0', '0', '1', '0', 'bob', '1']), np.array([0, 0, 1, 0, 2, 1]), criterion='entropy')
    # Information gain between ['0' '0' '1' '0' 'bob' '1'] and [0 0 1 0 2 1] is 0.0
    # with criterion 'gini' and 0.0 with criterion 'entropy'
