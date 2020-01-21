#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, theta):
        """
        Description:
        generator of the class, initialize self.
        Args:
        theta: has to be a list or a numpy array, it is a vector of
        dimension (number of features + 1, 1).
        Raises:
        This method should noot raise any Exception.
        """
        self.theta = np.asarray(theta)

    def cost_elem_(self, X, Y):
        """
        Description:
        Calculates all the elements 0.5/M*(y_pred - y)^2 of the cost
        function.
        Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of
        features + 1, 1).
        X: has to be a numpy.ndarray, a matrix of dimension (number of
        training examples, number of features).
        Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training
        examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        Raises:
        This function should not raise any Exception.
        """
        if X.size == 0 or self.theta.size == 0 or Y.size == 0 or \
                (X.size != 0 and X.shape[1] + 1 != self.theta.shape[0]) or \
            X.shape[0] != Y.shape[0]:
            return None
        # on commence par rajouter une colonne x0 qui vaut 1
        X_1 = np.insert(X, 0, [1.], axis=1)
        Y_hat = np.dot(X_1, self.theta)
        m = X.shape[0]
        return 0.5 / m * (Y_hat - Y) ** 2


    def cost_(self, X, Y):
        """
        Description:
        Calculates the value of cost function.
        Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of
        features + 1, 1).
        X: has to be a numpy.ndarray, a vector of dimension (number of
        training examples, number of features).
        Returns:
        J_value : has to be a float.
        None if X does not match the dimension of theta.
        Raises:
        This function should not raise any Exception.
        """
        return self.cost_elem_(X, Y).sum()

    def fit_(self, X, Y, alpha, n_cycle):
        """
        Description:
        Performs a fit of Y(output) with respect to X.
        Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of
        features + 1, 1).
        X: has to be a numpy.ndarray, a matrix of dimension (number of
        training examples, number of features).
        Y: has to be a numpy.ndarray, a vector of dimension (number of
        training examples, 1).
        alpha: positive float
        n_cycle: positive integer
        Returns:
        new_theta: numpy.ndarray, a vector of dimension (number of the
        features +1,1).
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exception.
        """
        if X.size == 0 or self.theta.size == 0 or Y.size == 0 or \
                (X.size != 0 and X.shape[1] + 1 != self.theta.shape[0]) or \
            X.shape[0] != Y.shape[0]:
            return None
        # on commence par rajouter une colonne x0 qui vaut 1
        X_1 = np.insert(X, 0, [1.], axis=1)
        # 1/m * transpose(X) * (X * theta - Y)
        m = X.shape[0]
        for _ in range(n_cycle):
            # theta = theta - (alpha / m) * np.dot(np.transpose(X_1), (np.dot(X_1, theta) - Y))
            self.theta = self.theta - (alpha / m) * np.dot(np.transpose(X_1), self.predict_(X) - Y)
        return self.theta

    def predict_(self, X):
        """
        Description:
        Prediction of output using the hypothesis function (linear model).
        Args:
        theta: has to be a numpy.ndarray, a vector of dimension (number of
        features + 1, 1).
        X: has to be a numpy.ndarray, a matrix of dimension (number of
        training examples, number of features).
        Returns:
        pred: numpy.ndarray, a vector of dimension (number of the training
        examples,1).
        None if X does not match the dimension of theta.
        Raises:
        This function should not raise any Exception.
        """
        if X.size == 0 or self.theta.size == 0 or \
                (X.size != 0 and X.shape[1] + 1 != self.theta.shape[0]):
            return None
        # on commence par rajouter une colonne x0 qui vaut 1
        X_1 = np.insert(X, 0, [1.], axis=1)
        return np.dot(X_1, self.theta)



if __name__ == '__main__':
    import numpy as np
    from mylinearregression import MyLinearRegression as MyLR
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLR([[1.], [1.], [1.], [1.], [1]])
    print(mylr.predict_(X))
    # => array([[8.], [48.], [323.]])
    print(mylr.cost_elem_(X, Y))
    # => array([[37.5], [0.], [1837.5]])
    print(mylr.cost_(X, Y))
    # => 1875.0
    mylr.fit_(X, Y, alpha=1.6e-4, n_cycle=200000)
    print(mylr.theta)
    # => array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])
    print(mylr.predict_(X))
    # => array([[23.499..], [47.385..], [218.079...]])
    print(mylr.cost_elem_(X, Y))
    # => array([[0.041..], [0.062..], [0.001..]])
    print(mylr.cost_(X, Y))
    # => 0.1056..
