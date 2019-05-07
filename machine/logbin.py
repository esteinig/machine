import numpy as np
from steinet.base import logistic, binary_log_loss, ClassifierMixin


class LogisticClassifierBinary(ClassifierMixin):
    """
    Logistic regression classifier for binary targets

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

        n_features = X.shape[1]
        self.w = np.random.normal(size=(n_features, ))
        self.b = np.random.normal(size=1)

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

        z = np.dot(X, self.w) + self.b
        y_hat = logistic(z)
        return y_hat

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

        grad_w = np.dot(X.T, delta)
        grad_b = np.sum(delta, axis=0)
        return grad_w, grad_b

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
        self.w_velocity = self.momentum * self.w_velocity - self.learning_rate * grad_w
        self.b_velocity = self.momentum * self.b_velocity - self.learning_rate * grad_b

        self.w += self.w_velocity
        self.b += self.b_velocity

        self.w -= self.lam1 * self.w + self.lam2 * np.sign(self.w)
        self.b -= self.lam1 * self.b + self.lam2 * np.sign(self.b)

        # self.w -= self.learning_rate * grad_w
        # self.b -= self.learning_rate * grad_b
        #
        # self.w -= self.lam1 * self.w + self.lam2 * np.sign(self.w)
        # self.b -= self.lam2 * self.b + self.lam2 * np.sign(self.b)

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

        if not hasattr(self, 'w'):
            self._initialize_weights(X)
        if self.w_velocity is None:
            self.w_velocity = np.zeros_like(self.w)
            self.b_velocity = np.zeros_like(self.b)
        training_loss = []
        training_accuracy = []
        self.optimal_loss = float('inf')
        print('Training in progress... ')
        for it in range(self.max_iter):
            if 10*it % self.max_iter == 0:
                chunk = int(10*it/self.max_iter)
                print('[' + '%'*chunk + '-'*(10 - chunk) + ']  ' + ''.join([str(10*chunk), '% complete']))
            y_hat = self._forward_pass(X)
            loss = binary_log_loss(y, y_hat)
            delta = (y_hat - y)/y.shape[0]
            grad_w, grad_b = self._compute_loss_grad(X, delta)
            self._update_params(grad_w, grad_b)
            training_loss.append(loss)

            y_pred = np.round(y_hat, 0)
            accuracy = self.score(y, y_pred)
            training_accuracy.append(accuracy)
            if loss < self.optimal_loss:
                self.optimal_w = self.w
                self.optimal_b = self.b
                self.optimal_loss = loss

        self.w = self.optimal_w
        self.b = self.optimal_b
        print('[' + '%' * 10 + ']  ' + ''.join([str(100), '% complete']))
        print('Training complete')
        return training_loss, training_accuracy

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

        y_hat = self._forward_pass(X)
        y_pred = np.round(y_hat, 0)
        return y_pred
