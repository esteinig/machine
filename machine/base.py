"""Utilities for the neural network modules
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy


def identity(X):
    """Simply return the input array.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.
    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return X


def logistic(X):
    """Compute the logistic function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return logistic_sigmoid(X, out=X)


def tanh(X):
    """Compute the hyperbolic tan function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return np.tanh(X, out=X)


def relu(X):
    """Compute the rectified linear unit function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):
    """Compute the K-way softmax function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu, 'softmax': softmax}


def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do


def inplace_logistic_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.
    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.
    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0

def logistic_derivative(Z):
    """
    Apply the derivative of the logistic sigmoid function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.
    """
    return np.exp(-Z)/(1 + np.exp(-Z))**2

def relu_derivative(Z):
    """
    Apply the derivative of the rectified linear unit function

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.
    """
    deriv = np.copy(Z)
    deriv[deriv <= 0] = 0
    deriv[deriv > 0] = 1
    return deriv

def tanh_derivative(Z):
    """
    Apply the derivative of the hyperbolic tangent function

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.
    """
    return (1/np.cosh(Z))**2



DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': tanh_derivative,
               'logistic': logistic_derivative,
               'relu': relu_derivative}


def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.
    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return ((y_true - y_pred) ** 2).mean() / 2


def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return - xlogy(y_true, y_prob).sum() / y_prob.shape[0]


def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.
    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return -(xlogy(y_true, y_prob) +
             xlogy(1 - y_true, 1 - y_prob)).sum() / y_prob.shape[0]


def one_hot(y, n_targets=None):
    n_samples = y.shape[0]
    if n_targets is None:
        n_targets = len(set(y))
    y_one_hot = np.zeros(shape=(n_samples, n_targets))
    for i in range(n_samples):
        y_one_hot[i, int(y[i])] = 1
    return y_one_hot


def batch_split(X, y, batch_size):
    n_samples = X.shape[0]
    idx = np.random.permutation(n_samples)
    X = X[idx, :]
    y = y[idx]
    batches = []
    for i in range(0, n_samples, batch_size):
        Xb = X[i:i + batch_size, :]
        yb = y[i:i + batch_size]
        batches.append(tuple(Xb, yb))
    return batches


class ClassifierMixin:
    """
    A classifier mixin object that supplies regular classifiers (e.g. logistic, MLP)
    with additional methods to summarize and visualize performance metrics

    ...

    Methods
    -------
    misclassified(y_true, y_pred)
        Compares predicted targets (y_pred) with actual targets (y_true) and returns the
        indices of misclassified samples
    score(y_true, y_pred)
        Compares predicted targets (y_pred) with actual targets (y_true) and returns the
        mean accuracy of the classifier
    plot_fit(X, y_true, full_grid=True)
        Produces a plot comparing the predicted targets y_pred associated with the dataset
        X and the actual targets y_true

    """

    def misclassified(self, y_true, y_pred):
        """
        Locate misclassified samples

        Parameters
        ----------
        y_true : array-like or label indicator matrix
            Ground truth (correct) labels.
        y_pred : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier's
            predict_proba method.

        Returns
        -------
        misidx : array-like
            Indices of correctly classified samples

        """
        mis_idx = y_true != y_pred
        return mis_idx

    def score(self, y_true, y_pred):
        """
        Compute the classification accuracy

        Parameters
        ----------
        y_true : array-like or label indicator matrix
            Ground truth (correct) labels.
        y_pred : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier's
            predict_proba method.

        Returns
        -------
        accuracy : float
            Mean number of correctly classified samples

        """
        return np.mean(y_pred == y_true)

    def plot_fit(self, X, y_true, visible_features=None, full_grid=True):
        """
        Plot the model fit

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y_true : array-like or label indicator matrix
            Ground truth (correct) labels.

        """
        n_features = X.shape[1]
        y_pred = self.predict(X)
        ax = None
        if visible_features is None:
            visible_features = [0, 1]
        if len(visible_features) == 2:
            plt.scatter(X[:, visible_features[0]], X[:, visible_features[1]],
                        c=y_true, edgecolors='w', linewidths=0.7)
        elif len(visible_features) == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            plt.scatter(X[:, visible_features[0]], X[:, visible_features[1]], X[:, visible_features[2]],
                        c=y_true, edgecolors='w', linewidths=0.5)
        if full_grid and n_features == len(visible_features):
            self._plot_grid(X, ax)
        else:
            mis_idx = y_pred != y_true
            if sum(mis_idx) > 0:
                if len(visible_features) == 2:
                    plt.scatter(X[mis_idx, visible_features[0]], X[mis_idx, visible_features[1]],
                                c=y_pred[mis_idx], marker='x', alpha=0.5)
                elif len(visible_features) == 3:
                    plt.scatter(X[mis_idx, visible_features[0]], X[mis_idx, visible_features[1]],
                                X[:, visible_features[2]],
                                c=y_pred[mis_idx], marker='x', alpha=0.5)
        plt.grid(linewidth=0.2)
        plt.show()

    def _generate_grid(self, X):
        n_features = X.shape[1]
        x_grids = []
        for feature in range(n_features):
            x_grids += [np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), 100)]
        if n_features == 2:
            X_grid = np.array([(x0, x1) for x0 in x_grids[0] for x1 in x_grids[1]])
        if n_features == 3:
            X_grid = np.array([(x0, x1, x2) for x0 in x_grids[0] for x1 in x_grids[1] for x2 in x_grids[2]])
        return X_grid

    def _plot_grid(self, X, ax=None):
        n_features = X.shape[1]
        X_grid = self._generate_grid(X)
        y_grid = self.predict(X_grid)
        if n_features == 2:
            plt.scatter(X_grid[:, 0], X_grid[:, 1], c=y_grid, alpha=0.05)
        if n_features == 3:
            ax.scatter(X_grid[:, 0], X_grid[:, 1], X_grid[:, 2], c=y_grid, alpha=0.05)


# class LogisticClassifierBinaryMixin:
#
#     def plot_fit(self, X, y, full_grid=True):
#         n_samples, n_features = X.shape
#         y_pred = self.predict(X)
#         mis_idx = self.misclassified(y, y_pred)
#         if n_features == 2:
#             grid = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 2)
#             m = -self.w[0]/self.w[1]
#             c = -self.b/self.w[1]
#             y_boundary = m*grid + c
#             plt.plot(grid, y_boundary)
#             plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='w', linewidths=0.5)
#             if full_grid:
#                 x0_grid = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
#                 x1_grid = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
#                 X_lattice = np.array([(x0, x1)
#                                    for x0 in x0_grid
#                                    for x1 in x1_grid])
#                 y_lattice = self.predict(X_lattice)
#                 plt.scatter(X_lattice[:, 0], X_lattice[:, 1], c=y_lattice, alpha=0.03)
#         elif n_features == 3:
#             plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolors='w', linewidths=0.5)
#             plt.scatter(X[mis_idx, 0], X[mis_idx, 1], X[mis_idx, 2], c=y_pred[mis_idx], marker='x')
#             if full_grid:
#                 x0_grid = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
#                 x1_grid = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
#                 x2_grid = np.linspace(np.min(X[:, 2]), np.max(X[:, 2]), 100)
#                 X_lattice = np.array([(x0, x1, x2)
#                                       for x0 in x0_grid
#                                       for x1 in x1_grid
#                                       for x2 in x2_grid])
#                 y_lattice = self.predict(X_lattice)
#                 fig = plt.figure()
#                 ax = Axes3D(fig)
#                 ax.scatter(X_lattice[:, 0], X_lattice[:, 1], X_lattice[:, 2], c=y_lattice, alpha=0.03)
#         plt.show()
#
#
# class LogisticClassifierMultinomialMixin:
#
#     def plot_fit(self, X, y, full_grid=True):
#         n_samples, n_features = X.shape
#         y_pred = self.predict(X)
#         mis_idx = self.misclassified(y, y_pred)
#         if n_features == 2:
#             plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='w', linewidths=0.7)
#             if full_grid:
#                 x0_grid = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
#                 x1_grid = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
#                 X_lattice = np.array([(x0, x1)
#                                    for x0 in x0_grid
#                                    for x1 in x1_grid])
#                 y_lattice = self.predict(X_lattice)
#                 plt.scatter(X_lattice[:, 0], X_lattice[:, 1], c=y_lattice, alpha=0.05)
#         elif n_features == 3:
#             plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolors='w', linewidths=0.5)
#             plt.scatter(X[mis_idx, 0], X[mis_idx, 1], X[mis_idx, 2], c=y_pred[mis_idx], marker='x')
#             if full_grid:
#                 x0_grid = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
#                 x1_grid = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
#                 x2_grid = np.linspace(np.min(X[:, 2]), np.max(X[:, 2]), 100)
#                 X_lattice = np.array([(x0, x1, x2)
#                                       for x0 in x0_grid
#                                       for x1 in x1_grid
#                                       for x2 in x2_grid])
#                 y_lattice = self.predict(X_lattice)
#                 fig = plt.figure()
#                 ax = Axes3D(fig)
#                 ax.scatter(X_lattice[:, 0], X_lattice[:, 1], X_lattice[:, 2], c=y_lattice, alpha=0.03)
#         plt.show()
