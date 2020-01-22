#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y_true: a scalar or a list for the correct labels
    y_pred: a scalar or a list for the predicted labels
    m: the length of y_true (should also be the length of y_pred)
    eps: eps (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    y_t, y_p = np.array(y_true), np.array(y_pred)
    cost = (1 / m) * (((-y_t).T * np.log(y_p + eps)) - ((1 - y_t).T * np.log(1 - y_p + eps)))
    # cost = (1 / m) * (((-y_t).T @ np.log(y_p + eps)) - ((1 - y_t).T @ np.log(1 - y_p + eps)))
    return cost if isinstance(cost, float) else cost.sum()


if __name__ == '__main__':
    # from sigmoid import sigmoid_
    # from log_loss import vec_log_loss_

    def sigmoid_(x):
        """
        Compute the sigmoid of a scalar or a list.
        Args:
        x: a scalar or list
        Returns:
        The sigmoid value as a scalar or list.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        x = np.asarray(x)
        sigm = 1. / (1. + np.exp(-x))
        return sigm
    # Test n.1
    x = 4
    y_true = 1
    theta = 0.5
    y_pred = sigmoid_(x * theta)
    m = 1  # length of y_true is 1
    print(vec_log_loss_(y_true, y_pred, m))
    # 0.12692801104297152
    # Test n.2
    x = [1, 2, 3, 4]
    y_true = 0
    theta = [-1.5, 2.3, 1.4, 0.7]
    x_dot_theta = sum([a * b for a, b in zip(x, theta)])
    y_pred = sigmoid_(x_dot_theta)
    m = 1
    print(vec_log_loss_(y_true, y_pred, m))
    # 10.100041078687479
    # Test n.3
    x_new = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    y_true = [1, 0, 1]
    theta = [-1.5, 2.3, 1.4, 0.7]
    x_dot_theta = []
    for i in range(len(x_new)):
        my_sum = 0
        for j in range(len(x_new[i])):
            my_sum += x_new[i][j] * theta[j]
        x_dot_theta.append(my_sum)
    y_pred = sigmoid_(x_dot_theta)
    m = len(y_true)
    print(vec_log_loss_(y_true, y_pred, m))
    # 7.233346147374828

    # Test n.1
    x = 4
    y_true = 1
    theta = 0.5
    y_pred = sigmoid_(x * theta)
    m = 1  # length of y_true is 1
    print(vec_log_loss_(y_true, y_pred, m))
    # 0.12692801104297152
    # Test n.2
    x = np.array([1, 2, 3, 4])
    y_true = 0
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    y_pred = sigmoid_(np.dot(x, theta))
    m = 1
    print(vec_log_loss_(y_true, y_pred, m))
    # 10.100041078687479
    # Test n.3
    x_new = np.arange(1, 13).reshape((3, 4))
    y_true = np.array([1, 0, 1])
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    y_pred = sigmoid_(np.dot(x_new, theta))
    m = len(y_true)
    print(vec_log_loss_(y_true, y_pred, m))
    # 7.233346147374828