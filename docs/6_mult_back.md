# Multinomial backpropagation

Last week we generalized our binary feed-forward equations to the multinomial (i.e. $$T>2$$) case. It remains now to determine the backpropagation equations so that we can update our weights and biases and improve the model accuracy, i.e. so that we can "train" our model.

The steps we follow here are almost identical to those presented in the binary logistic classification case so I encourage everyone to attempt the derivation of the backpropagation equations on their own before looking through the results presented here.

Once again to evaluate the full derivative we will invoke the chain rule

$$
\frac{\partial J}{\partial w_{lm}} &= \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T \frac{\partial J}{\partial \hat y_{ik}}\frac{\partial \hat y_{ik}}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial w_{lm}}
$$

and look at each of the pieces separately. Beginning first with the loss function we find

$$
\begin{aligned}
J &= -\frac{1}{N}\sum_{i=1}^N\sum_{k=1}^T y_{ik} \log \hat y_{ik}\\
\Rightarrow \quad \frac{\partial J}{\partial \hat y_{ik}} &= -\frac{1}{N} \frac{y_{ik}}{\hat y_{ik}}
\end{aligned}
$$

Next we turn to the output probabilities $$\hat y_{ik}$$:

$$
\begin{aligned}
\hat y_{ik} &= \frac{\mathrm{exp}(z_{ik})}{\sum_{k=1}^T \mathrm{exp}(z_{ik})} \\
\Rightarrow \frac{\partial \hat y_{ik}}{\partial z_{ij}} &= \frac{\left(\sum_{k'=1}^T \mathrm{exp}(z_{ik'})\right) \mathrm{exp}(z_{ik})\bar\delta_{kj} - \mathrm{exp}(z_{ik})\mathrm{exp}(z_{ij})}{\left(\sum_{k=1}^T \mathrm{exp}(z_{ik})\right)^2 \\
&= \frac{\mathrm{exp}(z_{ik})}{\sum_{k=1}^T \mathrm{exp}(z_{ik})}\left[\bar \delta_{kj} - \frac{\mathrm{exp}(z_{ij})}{\sum_{k=1}^T \mathrm{exp}(z_{ik})}\right]\\
&= \hat y_{ik}\left[\bar \delta_{kj} - \hat y_{ij}\right]
\end{aligned}
$$

where we have had to use $$\bar\delta_{kj}$$ to denote the Kronecker-delta symbol to distinguish it from the mean error $$\delta_{ik}$$.

Finally, we calculate the gradient of the activation $$z_{ij}$$ relative to the weights $$w_{lm}$$:

$$
\begin{aligned}
z_{ij} &= \sum_{r = 1}^D x_{ir} w_{rj} + b_j\\
\Rightarrow \frac{\partial z_{ij}}{\partial w_{lm}} &= x_{il}\bar\delta_{jm}.
\end{aligned}
$$

Putting it all together yields

$$
\begin{aligned}
\frac{\partial J}{\partial w_{lm}} &= \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T \frac{\partial J}{\partial \hat y_{ik}}\frac{\partial \hat y_{ik}}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial w_{lm}}\\
&= -\frac{1}{N} \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T \frac{y_{ik}}{\hat y_{ik}} \cdot \hat y_{ik}\left[\bar \delta_{kj} - \hat y_{ij}\right] \cdot x_{il}\bar\delta_{jm}\\
&= -\frac{1}{N} \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T y_{ik} \left[\bar \delta_{kj} - \hat y_{ij}\right] \cdot x_{il}\bar\delta_{jm}.
\end{aligned}
$$

First, let's collapse the sum over $$j$$ using the $$\bar\delta_{jm}$$ term. In this case all terms with a $$j$$ subscript will have the $$j$$ replaced by an $$m$$:

$$
\frac{\partial J}{\partial w_{lm}} =  -\frac{1}{N} \sum_{i=1}^N\sum_{k=1}^T y_{ik}\left[\bar\delta_{km} - \hat y_{im}\right]x_{il}.
$$

Next, we evaluate the summation over $$k$$. Notice that only the $$\bar \delta_{km}$$ depends on the index $$k$$, hence, expanding this summation we find

$$
\frac{\partial J}{\partial w_{lm}} = \frac{1}{N}\sum_{i = 1}^N (\hat y_{im} - y_{im})x_{il}
$$

Rewriting this expression in vector form gives

$$ 
\begin{equation}
\nabla_w J = \frac{1}{N} X^T(\hat y - y) = X^T 
\delta\label{eq:nabla_w}
\tag{1}
\end{equation}
$$

where we have once again introduced the mean error term $$\delta$$ (which has now been generalized to a matrix rather than a vector):

$$
\delta_{ik} = \frac{(\hat y_{ik} - y_{ik})}{N}.
$$

Following this calculation, it is straightforward to determine the gradient of the loss function $$J$$ with respect to the bias terms $$b_k$$:

$$
\frac{\partial J}{\partial b_{m}} = \frac{1}{N} \sum_{i=1}^N (\hat y_{im} - y_{im})
$$

which in vector form becomes

$$
\begin{equation}
\nabla_b J = \mathbf{1}^T\delta.
\label{eq:nabla_b}
\tag{2}
\end{equation}
$$

Note that the vector form of the equations of the gradients $$\nabla_w$$ and $$\nabla_b$$ are the general expression used independent of the activation or loss functions chosen --- the variable that is modified in general is the definition of the mean error term $$\delta$$:

$$
\delta  = \nabla_{\hat y} J \odot \sigma'(z).
$$

Here $$\nabla_{\hat y} J$$ is the derivative of the loss function with respect to the output probabilities $$\hat y$$ and $$\sigma '(z)$$ is the derivative of the activation function $$\sigma(z)$$ with respect to the activations $$z$$.

Once again, for convenience, I have summarized the pythonic version of the feed-forward and backpropagation equations below (notice that they are almost identical to those given for the binary logistic classifier, we have just had to make one or two adjustments to account for the increased dimension):

#### Feed-forward

`z = np.dot(X, w) + b`

`y_hat = np.exp(z)/np.exp(z).sum(axis=1)`

`loss = -xlogy(y, y_hat).sum()/n_samples`

#### Backpropagation

`delta = (y_hat - y)/y.shape[0]`

`grad_w = np.dot(X.T, delta)`

`grad_b = np.sum(delta, axis=0)`

`w -= self.learning_rate * grad_w`

`b -= self.learning_rate * grad_b`



## Weekly tasks
This week we will update our `LogisticClassifierMultinomial` object by providing it with the ability to train by adding the method `fit()`.


#### `LogisticClassifierMultinomial`
- In `LogisticClassifierMultinomial` write an internal (private) method `_compute_loss_grad(X, delta)` that computes the gradient of loss with respect to the weights and biases
- Write another internal method `_update_params(grad_w, grad_b)` that updates the weights and biases based on the loss gradients
- Write a `fit(X, y)` method that accepts a sample dataset `X` and target set `y` and trains the model to minimize the loss function
	
Once you have written each of the methods above try your new model out on one of your toy datasets. Make sure to split the dataset into training and test sets: fitting the model to the training set and then validating the model fit on the test set. For which datasets does the logistic classifier work best and for which does it provide a poor fit?
