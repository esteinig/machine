# Multinomial backpropagation

Last week we generalized our binary feed-forward equations to the multinomial (i.e. <img src="https://tex.s2cms.ru/svg/T%3E2" alt="T&gt;2" />) case. It remains now to determine the backpropagation equations so that we can update our weights and biases and improve the model accuracy, i.e. so that we can "train" our model.

The steps we follow here are almost identical to those presented in the binary logistic classification case so I encourage everyone to attempt the derivation of the backpropagation equations on their own before looking through the results presented here.

Once again to evaluate the full derivative we will invoke the chain rule

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_%7Blm%7D%7D%20%26%3D%20%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bj%3D1%7D%5ED%5Csum_%7Bk%3D1%7D%5ET%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_%7Bik%7D%7D%5Cfrac%7B%5Cpartial%20%5Chat%20y_%7Bik%7D%7D%7B%5Cpartial%20z_%7Bij%7D%7D%5Cfrac%7B%5Cpartial%20z_%7Bij%7D%7D%7B%5Cpartial%20w_%7Blm%7D%7D%0A" alt="
\frac{\partial J}{\partial w_{lm}} &amp;= \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T \frac{\partial J}{\partial \hat y_{ik}}\frac{\partial \hat y_{ik}}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial w_{lm}}
" />

and look at each of the pieces separately. Beginning first with the loss function we find

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0AJ%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bk%3D1%7D%5ET%20y_%7Bik%7D%20%5Clog%20%5Chat%20y_%7Bik%7D%5C%5C%0A%5CRightarrow%20%5Cquad%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_%7Bik%7D%7D%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Cfrac%7By_%7Bik%7D%7D%7B%5Chat%20y_%7Bik%7D%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
J &amp;= -\frac{1}{N}\sum_{i=1}^N\sum_{k=1}^T y_{ik} \log \hat y_{ik}\\
\Rightarrow \quad \frac{\partial J}{\partial \hat y_{ik}} &amp;= -\frac{1}{N} \frac{y_{ik}}{\hat y_{ik}}
\end{aligned}
" />

Next we turn to the output probabilities <img src="https://tex.s2cms.ru/svg/%5Chat%20y_%7Bik%7D" alt="\hat y_{ik}" />:

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Chat%20y_%7Bik%7D%20%26%3D%20%5Cfrac%7B%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%7B%5Csum_%7Bk%3D1%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%20%5C%5C%0A%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%5Chat%20y_%7Bik%7D%7D%7B%5Cpartial%20z_%7Bij%7D%7D%20%26%3D%20%5Cfrac%7B%5Cleft(%5Csum_%7Bk'%3D1%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik'%7D)%5Cright)%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%5Cbar%5Cdelta_%7Bkj%7D%20-%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%5Cmathrm%7Bexp%7D(z_%7Bij%7D)%7D%7B%5Cleft(%5Csum_%7Bk%3D1%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%5Cright)%5E2%20%5C%5C%0A%26%3D%20%5Cfrac%7B%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%7B%5Csum_%7Bk%3D1%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%5Cleft%5B%5Cbar%20%5Cdelta_%7Bkj%7D%20-%20%5Cfrac%7B%5Cmathrm%7Bexp%7D(z_%7Bij%7D)%7D%7B%5Csum_%7Bk%3D1%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%5Cright%5D%5C%5C%0A%26%3D%20%5Chat%20y_%7Bik%7D%5Cleft%5B%5Cbar%20%5Cdelta_%7Bkj%7D%20-%20%5Chat%20y_%7Bij%7D%5Cright%5D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\hat y_{ik} &amp;= \frac{\mathrm{exp}(z_{ik})}{\sum_{k=1}^T \mathrm{exp}(z_{ik})} \\
\Rightarrow \frac{\partial \hat y_{ik}}{\partial z_{ij}} &amp;= \frac{\left(\sum_{k'=1}^T \mathrm{exp}(z_{ik'})\right) \mathrm{exp}(z_{ik})\bar\delta_{kj} - \mathrm{exp}(z_{ik})\mathrm{exp}(z_{ij})}{\left(\sum_{k=1}^T \mathrm{exp}(z_{ik})\right)^2 \\
&amp;= \frac{\mathrm{exp}(z_{ik})}{\sum_{k=1}^T \mathrm{exp}(z_{ik})}\left[\bar \delta_{kj} - \frac{\mathrm{exp}(z_{ij})}{\sum_{k=1}^T \mathrm{exp}(z_{ik})}\right]\\
&amp;= \hat y_{ik}\left[\bar \delta_{kj} - \hat y_{ij}\right]
\end{aligned}
" />

where we have had to use <img src="https://tex.s2cms.ru/svg/%5Cbar%5Cdelta_%7Bkj%7D" alt="\bar\delta_{kj}" /> to denote the Kronecker-delta symbol to distinguish it from the mean error <img src="https://tex.s2cms.ru/svg/%5Cdelta_%7Bik%7D" alt="\delta_{ik}" />.

Finally, we calculate the gradient of the activation <img src="https://tex.s2cms.ru/svg/z_%7Bij%7D" alt="z_{ij}" /> relative to the weights <img src="https://tex.s2cms.ru/svg/w_%7Blm%7D" alt="w_{lm}" />:

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0Az_%7Bij%7D%20%26%3D%20%5Csum_%7Br%20%3D%201%7D%5ED%20x_%7Bir%7D%20w_%7Brj%7D%20%2B%20b_j%5C%5C%0A%5CRightarrow%20%5Cfrac%7B%5Cpartial%20z_%7Bij%7D%7D%7B%5Cpartial%20w_%7Blm%7D%7D%20%26%3D%20x_%7Bil%7D%5Cbar%5Cdelta_%7Bjm%7D.%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
z_{ij} &amp;= \sum_{r = 1}^D x_{ir} w_{rj} + b_j\\
\Rightarrow \frac{\partial z_{ij}}{\partial w_{lm}} &amp;= x_{il}\bar\delta_{jm}.
\end{aligned}
" />

Putting it all together yields

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_%7Blm%7D%7D%20%26%3D%20%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bj%3D1%7D%5ED%5Csum_%7Bk%3D1%7D%5ET%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_%7Bik%7D%7D%5Cfrac%7B%5Cpartial%20%5Chat%20y_%7Bik%7D%7D%7B%5Cpartial%20z_%7Bij%7D%7D%5Cfrac%7B%5Cpartial%20z_%7Bij%7D%7D%7B%5Cpartial%20w_%7Blm%7D%7D%5C%5C%0A%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bj%3D1%7D%5ED%5Csum_%7Bk%3D1%7D%5ET%20%5Cfrac%7By_%7Bik%7D%7D%7B%5Chat%20y_%7Bik%7D%7D%20%5Ccdot%20%5Chat%20y_%7Bik%7D%5Cleft%5B%5Cbar%20%5Cdelta_%7Bkj%7D%20-%20%5Chat%20y_%7Bij%7D%5Cright%5D%20%5Ccdot%20x_%7Bil%7D%5Cbar%5Cdelta_%7Bjm%7D%5C%5C%0A%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bj%3D1%7D%5ED%5Csum_%7Bk%3D1%7D%5ET%20y_%7Bik%7D%20%5Cleft%5B%5Cbar%20%5Cdelta_%7Bkj%7D%20-%20%5Chat%20y_%7Bij%7D%5Cright%5D%20%5Ccdot%20x_%7Bil%7D%5Cbar%5Cdelta_%7Bjm%7D.%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial J}{\partial w_{lm}} &amp;= \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T \frac{\partial J}{\partial \hat y_{ik}}\frac{\partial \hat y_{ik}}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial w_{lm}}\\
&amp;= -\frac{1}{N} \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T \frac{y_{ik}}{\hat y_{ik}} \cdot \hat y_{ik}\left[\bar \delta_{kj} - \hat y_{ij}\right] \cdot x_{il}\bar\delta_{jm}\\
&amp;= -\frac{1}{N} \sum_{i=1}^N\sum_{j=1}^D\sum_{k=1}^T y_{ik} \left[\bar \delta_{kj} - \hat y_{ij}\right] \cdot x_{il}\bar\delta_{jm}.
\end{aligned}
" />

First, let's collapse the sum over <img src="https://tex.s2cms.ru/svg/j" alt="j" /> using the <img src="https://tex.s2cms.ru/svg/%5Cbar%5Cdelta_%7Bjm%7D" alt="\bar\delta_{jm}" /> term. In this case all terms with a <img src="https://tex.s2cms.ru/svg/j" alt="j" /> subscript will have the <img src="https://tex.s2cms.ru/svg/j" alt="j" /> replaced by an <img src="https://tex.s2cms.ru/svg/m" alt="m" />:

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_%7Blm%7D%7D%20%3D%20%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bk%3D1%7D%5ET%20y_%7Bik%7D%5Cleft%5B%5Cbar%5Cdelta_%7Bkm%7D%20-%20%5Chat%20y_%7Bim%7D%5Cright%5Dx_%7Bil%7D.%0A" alt="
\frac{\partial J}{\partial w_{lm}} =  -\frac{1}{N} \sum_{i=1}^N\sum_{k=1}^T y_{ik}\left[\bar\delta_{km} - \hat y_{im}\right]x_{il}.
" />

Next, we evaluate the summation over <img src="https://tex.s2cms.ru/svg/k" alt="k" />. Notice that only the <img src="https://tex.s2cms.ru/svg/%5Cbar%20%5Cdelta_%7Bkm%7D" alt="\bar \delta_{km}" /> depends on the index <img src="https://tex.s2cms.ru/svg/k" alt="k" />, hence, expanding this summation we find

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_%7Blm%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%20%3D%201%7D%5EN%20(%5Chat%20y_%7Bim%7D%20-%20y_%7Bim%7D)x_%7Bil%7D%0A" alt="
\frac{\partial J}{\partial w_{lm}} = \frac{1}{N}\sum_{i = 1}^N (\hat y_{im} - y_{im})x_{il}
" />

Rewriting this expression in vector form gives

<img src="https://tex.s2cms.ru/svg/%20%0A%5Cbegin%7Bequation%7D%0A%5Cnabla_w%20J%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20X%5ET(%5Chat%20y%20-%20y)%20%3D%20X%5ET%20%0A%5Cdelta%5Clabel%7Beq%3Anabla_w%7D%0A%5Ctag%7B1%7D%0A%5Cend%7Bequation%7D%0A" alt=" 
\begin{equation}
\nabla_w J = \frac{1}{N} X^T(\hat y - y) = X^T 
\delta\label{eq:nabla_w}
\tag{1}
\end{equation}
" />

where we have once again introduced the mean error term <img src="https://tex.s2cms.ru/svg/%5Cdelta" alt="\delta" /> (which has now been generalized to a matrix rather than a vector):

<img src="https://tex.s2cms.ru/svg/%0A%5Cdelta_%7Bik%7D%20%3D%20%5Cfrac%7B(%5Chat%20y_%7Bik%7D%20-%20y_%7Bik%7D)%7D%7BN%7D.%0A" alt="
\delta_{ik} = \frac{(\hat y_{ik} - y_{ik})}{N}.
" />

Following this calculation, it is straightforward to determine the gradient of the loss function <img src="https://tex.s2cms.ru/svg/J" alt="J" /> with respect to the bias terms <img src="https://tex.s2cms.ru/svg/b_k" alt="b_k" />:

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b_%7Bm%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20(%5Chat%20y_%7Bim%7D%20-%20y_%7Bim%7D)%0A" alt="
\frac{\partial J}{\partial b_{m}} = \frac{1}{N} \sum_{i=1}^N (\hat y_{im} - y_{im})
" />

which in vector form becomes

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Bequation%7D%0A%5Cnabla_b%20J%20%3D%20%5Cmathbf%7B1%7D%5ET%5Cdelta.%0A%5Clabel%7Beq%3Anabla_b%7D%0A%5Ctag%7B2%7D%0A%5Cend%7Bequation%7D%0A" alt="
\begin{equation}
\nabla_b J = \mathbf{1}^T\delta.
\label{eq:nabla_b}
\tag{2}
\end{equation}
" />

Note that the vector form of the equations of the gradients <img src="https://tex.s2cms.ru/svg/%5Cnabla_w" alt="\nabla_w" /> and <img src="https://tex.s2cms.ru/svg/%5Cnabla_b" alt="\nabla_b" /> are the general expression used independent of the activation or loss functions chosen --- the variable that is modified in general is the definition of the mean error term <img src="https://tex.s2cms.ru/svg/%5Cdelta" alt="\delta" />:

<img src="https://tex.s2cms.ru/svg/%0A%5Cdelta%20%20%3D%20%5Cnabla_%7B%5Chat%20y%7D%20J%20%5Codot%20%5Csigma'(z).%0A" alt="
\delta  = \nabla_{\hat y} J \odot \sigma'(z).
" />

Here <img src="https://tex.s2cms.ru/svg/%5Cnabla_%7B%5Chat%20y%7D%20J" alt="\nabla_{\hat y} J" /> is the derivative of the loss function with respect to the output probabilities <img src="https://tex.s2cms.ru/svg/%5Chat%20y" alt="\hat y" /> and <img src="https://tex.s2cms.ru/svg/%5Csigma%20'(z)" alt="\sigma '(z)" /> is the derivative of the activation function <img src="https://tex.s2cms.ru/svg/%5Csigma(z)" alt="\sigma(z)" /> with respect to the activations <img src="https://tex.s2cms.ru/svg/z" alt="z" />.

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
