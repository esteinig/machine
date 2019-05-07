import numpy as np
from steinet.base import identity, tanh, relu, logistic, softmax, \
    binary_log_loss, log_loss, squared_loss, \
    inplace_identity_derivative, tanh_derivative,\
    logistic_derivative, relu_derivative,\
    one_hot, ClassifierMixin


ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu, 'softmax': softmax}

LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}

DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': tanh_derivative,
               'logistic': logistic_derivative,
               'relu': relu_derivative}


class MLPClassifier(ClassifierMixin):

    def __init__(self, hidden_layer_sizes=(), activation='logistic',
                 out_activation='softmax', learning_rate=0.05,
                 max_iter=200, lam1=0.0, lam2=0.0, momentum=0.9):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.out_activation = out_activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lam1 = lam1
        self.lam2 = lam2
        self.momentum = momentum
        self.optimal_w = None
        self.optimal_b = None
        self.w_velocity = None
        self.b_velocity = None

    def _initialize_weights(self, layer_units):
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



    def _forward_pass(self, z, activations):
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



    def _compute_loss_grad(self, activation, delta):
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


    def _backprop(self, X, y, z, activations, deltas, grad_w, grad_b):
        """
        Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        grad_w : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        grad_b : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        grad_w : list, length = n_layers - 1
        grad_b : list, length = n_layers - 1
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


    def _initialize_activations(self, X):
        n_samples, n_features = X.shape
        layer_units = ([n_features] + list(self.hidden_layer_sizes) + [self.n_targets])
        activations = [X]
        activations.extend(np.empty((n_samples, n_fan_out))
                           for n_fan_out in layer_units[1:])
        z = [np.empty_like(a_layer) for a_layer in activations[1:]]
        return z, activations
