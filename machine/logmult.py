import numpy as np
from machine.base import softmax, log_loss, ClassifierMixin, one_hot


class LogisticClassifierMultinomial(ClassifierMixin):
    """
    Logistic regression classifier for multinomial targets

    Attributes
    ----------
    learning_rate :

    max_iter :

    lam1 :

    lam2 :

    w :

    b :

    optimal_w :

    optimal_b :

    Methods
    -------
    _initialize_weights(X)
        Randomly initialize all model weights prior to model fitting
    _forward_pass(X)
        Perform a forward pass on the network by computing the values
        of the output layer.
    _compute_loss_grad(X, delta)
        Compute the gradient of loss with respect to coefs and intercept for
        the input layer.
    _update_params(grad_w, grad_b)
        Update the model weights and biases given the coef grads
    fit(X, y)
        Fit the model to data matrix X and target(s) y.
    predict(X)
        Use the model to predict the target labels for a set of samples X
    """

    def __init__(self, learning_rate=0.05, max_iter=200,
                 lam1=0.0, lam2=0.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lam1 = lam1
        self.lam2 = lam2
        self.momentum = momentum
        self.optimal_w = None
        self.optimal_b = None
        self.w_velocity = None
        self.b_velocity = None

    def _initialize_weights(self, X):
        """
        Randomly initialize all model weights prior to model fitting

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The ith row of the array holds the values of the ith sample.

        Returns
        -------
        self : an untrained model with random weights
        """


    def _forward_pass(self, X):
        """
        Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The ith row of the array holds the values of the ith sample.

        Returns
        -------
        y_hat : array, shape = (n_samples, n_targets)
            Predicted probabilities for each target and sample.
        """

    def _compute_loss_grad(self, X, delta):
        """
        Compute the gradient of loss with respect to coefs and intercept for
        the input layer.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The ith row of the array holds the values of the ith sample.
        delta : array, shape = (n_samples, n_targets)
            Loss function derivative

        Returns
        -------
        grad_w : array, shape = (n_features, n_targets)
            The gradient of the loss function with respect to the weights
        grad_b : array, shape = (, n_targets)
            The gradient of the loss function with respect to the bias
        """


    def _update_params(self, grad_w, grad_b):
        """
        Update the model weights and biases given the coef grads

        Parameters
        ----------
        grad_w : array, shape = (n_features, n_targets)
            The gradient of the loss function with respect to the weights
        grad_b : array, shape = (, n_targets)
            The gradient of the loss function with respect to the bias

        Returns
        -------
        self : returns an updated model.
        """



    def fit(self, X, y):
        """
        Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """


    def predict(self, X):
        """
        Use the model to predict the target labels for a set of samples X

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        """

