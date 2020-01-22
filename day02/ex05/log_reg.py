#!/usr/bin/env python3
# -*-coding:utf-8 -*

import numpy as np

class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate # can be 'constant' or 'invscaling'
        self.thetas = []
        # Your code here (e.g. a list of loss for each epochs...)
        self.loss_list =[]
        self.alpha_list = []

    def fit(self, x_train, y_train):
        """
        Fit the model according to the given training data.
        Args:
        x_train: a 1d or 2d numpy ndarray for the samples
        y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
        self : object
        None on any error.
        Raises:
        This method should not raise any Exception.
        """
        # Your code here

    def predict(self, x_train):
        """
        Predict class labels for samples in x_train.
        Arg:
        x_train: a 1d or 2d numpy ndarray for the samples
        Returns:
        y_pred, the predicted class label per sample.
        None on any error.
        Raises:
        This method should not raise any Exception.
        """
        return np.dot(x_train, self.thetas)

    def score(self, x_train, y_train):
        """
        Returns the mean accuracy on the given test data and labels.
        Arg:
        x_train: a 1d or 2d numpy ndarray for the samples
        y_train: a scalar or a numpy ndarray for the correct labels
        Returns:
        Mean accuracy of self.predict(x_train) with respect to y_true
        None on any error.
        Raises:
        This method should not raise any Exception.
        """
        # Your code here

    def __sigmoid_(self, x):
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

    def __vec_log_gradient_(self, x, y_true, y_pred):
        """
        Compute the gradient.
        Args:
        x: a list or a matrix (list of lists) for the samples
        y_true: a scalar or a list for the correct labels
        y_pred: a scalar or a list for the predicted labels
        Returns:
        The gradient as a scalar or a list of the width of x.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        X = np.array(x)
        y_t, y_p = np.array(y_true), np.array(y_pred)
        # gradient = np.dot(X.T, (y_p - y_t)) / y_t.shape[0]
        # gradient = ((y_p - y_t) * X.T) / X.shape[0]
        gradient = np.dot((y_p - y_t), X)
        return gradient

    def __vec_log_loss_(self, y_true, y_pred, m, eps=1e-15):
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
    import pandas as pd
    import numpy as np
    # from log_reg import LogisticRegressionBatchGd

    # We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
    df_train = pd.read_csv('dataset/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
    x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
    df_test = pd.read_csv('dataset/test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
    x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]
    # We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
    model = LogisticRegressionBatchGd(alpha=0.01, max_iter=1500, verbose=True, learning_rate='constant')
    # We fit our model to our dataset and display the score for the train and test datasets
    model.fit(x_train, y_train)
    print(f'Score on train dataset : {model.score(x_train, y_train)}')
    y_pred = model.predict(x_test)
    print(f'Score on test dataset : {(y_pred == y_test).mean()}')
    # epoch 0: loss 2.711028065632692
    # epoch 150: loss 1.760555094793668
    # epoch 300: loss 1.165023422947427
    # epoch 450: loss 0.830808383847448
    # epoch 600: loss 0.652110347325305
    # epoch 750: loss 0.555867078788320
    # epoch 900: loss 0.501596689945403
    # epoch 1050: loss 0.469145216528238
    # epoch 1200: loss 0.448682476966280
    # epoch 1350: loss 0.435197719530431
    # epoch 1500: loss 0.425934034947101
    # Score on train dataset: 0.7591904425539756
    # Score on test dataset: 0.7637737239727289
    # This is an example with verbose set to True, you could choose to display
    # your loss at the epochs you want.
    # Here I choose to only display 11 rows no matter how many epochs I had.
    # Your score should be pretty close to mine.
    # Your loss may be quite different weither you choose different hyperparameters,
    # if you add an intercept to your x_train
    # or if you shuffle your x_train at each epochs (this introduce stochasticity !) etc...
    # You might not get a score as good as
    #     sklearn.linear_model.LogisticRegression because it uses a different algorithm and
    # more optimized parameters that would require more time to implement.