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
        X: has to be a numpy.ndarray, a matrix of dimension (number of
        training examples, number of features).
        Y: has to be a numpy.ndarray, a vector of dimension (number of
        training examples, 1).
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
        X: has to be a numpy.ndarray, a vector of dimension (number of
        training examples, number of features).
        Y: has to be a numpy.ndarray, a vector of dimension (number of
        training examples, 1).
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

    def mse_(self, Y_hat, Y):
        """Computes the mean squared error of two non-empty numpy.ndarray,
        after computing the estimated/predicted value of Y from X
        without any for loop. The two arrays must have the same dimensions.
        Args:
        Y_hat: has to be an numpy.ndarray, a vector.
        Y: has to be an numpy.ndarray, a vector.
        Returns:
        The mean squared error of the two vectors as a float.
        None if Y or Y_hat are empty numpy.ndarray.
        None if Y and Y_hat does not share the same dimensions.
        Raises:
        This function should not raise any Exception.
        """
        if Y_hat.size == 0 or Y.size == 0 or (Y.size != 0 and Y.shape[0] != Y_hat.shape[0]):
            return None
        m = Y_hat.shape[0]
        return np.dot(np.transpose(Y - Y_hat), (Y - Y_hat)).sum() / m


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    # from . import MyLinearRegression as MyLinearRegression

    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1,1)

    linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
    linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    # affichage graphique blue pills
    fig, ax = plt.subplots()
    ax.scatter(Xpill, Yscore)
    ax.scatter(Xpill, Y_model1, c="green")
    ax.scatter(Xpill, Y_model2, c="red")

    plt.xticks(rotation=90)
    plt.xlabel("Quantity of blue pills (in micrograms)")
    plt.ylabel("Space driving score")
    plt.title("Evolution of the space driving score in function of the blue pill's quantity (in µg)")
    plt.plot(Xpill, Y_model1, "--", c="green", label="model 1")
    plt.plot(Xpill, Y_model2, "-.", c="red", label="model 2")
    fig.legend(loc="lower left")
    plt.show()
    plt.cla()

    # affichage fonction coût en fonction de théta
    theta1 = np.array([[89.0], [-8]])
    C0 = linear_model1.cost_(Xpill, Y_model1)
    theta2 = linear_model1.fit_(Xpill, Yscore, alpha=1.6e-4, n_cycle=10000)
    plt.xticks(rotation=90)
    plt.xlabel("Theta1")
    plt.ylabel("Cost function J(Theta0, Theta1")
    plt.title("Evolution of the cost function J in fuction of Theta0 for different values of Theta1")
    plt.plot(Xpill, Y_model1, "--", c="green", label="model 1")
    plt.plot(Xpill, Y_model2, "-.", c="red", label="model 2")
    fig.legend(loc="lower left")
    plt.show()
    plt.cla()

    print(linear_model1.mse_(Y_model1, Yscore))
    # 57.60304285714282
    print(mean_squared_error(Yscore, Y_model1))
    # 57.603042857142825
    print(linear_model2.mse_(Y_model2, Yscore))
    # 232.16344285714285
    print(mean_squared_error(Yscore, Y_model2))
    # 232.16344285714285