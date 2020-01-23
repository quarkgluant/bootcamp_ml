#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

def accuracy_score_(y_true, y_pred):
    """
    Compute the accuracy score.
    Args:
    y_true: a scalar or a numpy ndarray for the correct labels
    y_pred: a scalar or a numpy ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    return (y_true == y_pred).mean()

if __name__ == '__main__':
    import numpy as np
    # from accuracy import accuracy_score_
    from sklearn.metrics import accuracy_score

    # Test n.1
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(accuracy_score_(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    # 0.5
    # 0.5

    # Test n.2
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print(accuracy_score_(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    # 0.625
    # 0.625