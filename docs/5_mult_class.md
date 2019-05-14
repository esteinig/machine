# Multinomial classification

* So far we have dealt with binary classification problems where the output predictions are either TRUE or FALSE, 1 or 0, etc
* The logistic classifier we have written can be extended to deal with more general classification problems where the number of target classes is arbitrary. We move then from drawing a single line (or plane) to separate our dataset into drawing several lines (or planes)

<a href='https://github.com/esteinig'><img 
src='docs/img/logbin_net.png' align="middle" height="210" 
/></a>

<a href='https://github.com/esteinig'><img 
src='docs/img/logbin_net.png' align="middle" height="210" 
/></a>

*To generalize to multiple output classes <img src="https://tex.s2cms.ru/svg/T" alt="T" />, we need a new set of weights for each target class <img src="https://tex.s2cms.ru/svg/k%5Cin%5C%7B0%2C1%2C2%2C%5Cldots%2CT%5C%7D" alt="k\in\{0,1,2,\ldots,T\}" />:

<img src="https://tex.s2cms.ru/svg/%20w_j%20%5Crightarrow%20w_%7Bjk%7D%20" alt=" w_j \rightarrow w_{jk} " />

and

<img src="https://tex.s2cms.ru/svg/%20b%20%5Crightarrow%20b_k%20" alt=" b \rightarrow b_k " />

* In practice this means we are moving from vectors to matrices 
* We then have

<img src="https://tex.s2cms.ru/svg/%20z_%7Bik%7D%20%3D%20%5Csum_%7Bj%3D1%7D%5ED%20x_%7Bij%7Dw_%7Bjk%7D%20%2B%20b_k%20" alt=" z_{ik} = \sum_{j=1}^D x_{ij}w_{jk} + b_k " />

* Note that the summation here is over the dimensions (or features) of the dataset

* To generalize to the case of several target we need to modify our sigmoid activation function (i.e. the output activation function). For this purpose we use the "softmax" function:

<img src="https://tex.s2cms.ru/svg/%20%5Chat%20y_%7Bik%7D%20%3D%20%5Cmathrm%7Bsoftmax%7D(z_%7Bik%7D)%20%3D%20%5Cfrac%7B%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%7B%5Csum_%7Bk%3D1%7D%5ET%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%20" alt=" \hat y_{ik} = \mathrm{softmax}(z_{ik}) = \frac{\mathrm{exp}(z_{ik})}{\sum_{k=1}^T\mathrm{exp}(z_{ik})} " />

* Notice that the softmax function maps the real line <img src="https://tex.s2cms.ru/svg/%5Cmathbb%7BR%7D" alt="\mathbb{R}" /> onto the interval <img src="https://tex.s2cms.ru/svg/(0%2C1)" alt="(0,1)" />
* Moreover, if we sum across the targets for each sample <img src="https://tex.s2cms.ru/svg/i" alt="i" /> we get

<img src="https://tex.s2cms.ru/svg/%20%5Csum_%7Bk%20%3D%201%7D%5ET%20%5Chat%20y_%7Bik%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bk%20%3D%201%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%7B%5Csum_%7Bk%20%3D%201%7D%5ET%20%5Cmathrm%7Bexp%7D(z_%7Bik%7D)%7D%20%3D%201%20" alt=" \sum_{k = 1}^T \hat y_{ik} = \frac{\sum_{k = 1}^T \mathrm{exp}(z_{ik})}{\sum_{k = 1}^T \mathrm{exp}(z_{ik})} = 1 " />

* Side note:
   * Given the properties of the <img src="https://tex.s2cms.ru/svg/%20%5Chat%20y_%7Bik%7D%20" alt=" \hat y_{ik} " /> (in that they are between <img src="https://tex.s2cms.ru/svg/0%20" alt="0 " /> and <img src="https://tex.s2cms.ru/svg/%201%20" alt=" 1 " /> and sum to <img src="https://tex.s2cms.ru/svg/%201%20" alt=" 1 " />) it is tempting to think them as probabilities
   * In reality however, they are merely pseudo-probabilities: beyond sharing the minimum basic requirements of probabilities they do not share the additional interpretation that if I observe a sample <img src="https://tex.s2cms.ru/svg/%20x%20" alt=" x " /> it will be assigned to the class <img src="https://tex.s2cms.ru/svg/%20k%20" alt=" k " /> with probability <img src="https://tex.s2cms.ru/svg/%5Chat%20y%20" alt="\hat y " /> (think in frequentist terms)
* Once we have calculated the output probabilities for sample and target we make predictions by taking the maximum argument in each row

<img src="https://tex.s2cms.ru/svg/%20%5Chat%20Y_i%20%3D%20%5Cmathrm%7Bargmax%7D_k%20(%5Chat%20y_%7Bik%7D)%20" alt=" \hat Y_i = \mathrm{argmax}_k (\hat y_{ik}) " />

##### Exercise: Check for yourself that for the case <img src="https://tex.s2cms.ru/svg/%20T%20%3D%202%20" alt=" T = 2 " /> (i.e. when there are only two target classes) the softmax function reduces to the sigmoid/logistic function

* As before, we would like to compare our output predictions <img src="https://tex.s2cms.ru/svg/%20%5Chat%20y_%7Bik%7D%20" alt=" \hat y_{ik} " /> with the actual target labels <img src="https://tex.s2cms.ru/svg/%20y_i%20" alt=" y_i " />
* Immediately we notice the dimension mismatch: we are trying to comare a matrix of output (pseudo-)probabilities <img src="https://tex.s2cms.ru/svg/%20%5Chat%20y_%7Bik%7D%20" alt=" \hat y_{ik} " /> with a vector of target labels <img src="https://tex.s2cms.ru/svg/%20y_i%20" alt=" y_i " />
* Consider the example below

<img src="https://tex.s2cms.ru/svg/%0A%5Chat%20y%20%3D%20%5Cleft%5B%0A%5Cbegin%7Bmatrix%7D%0A0.2%20%26%200.1%20%26%200.3%20%26%200.4%20%5C%5C%0A0.4%20%26%200.2%20%26%200.2%20%26%200.2%20%5C%5C%0A0.1%20%26%200.5%20%26%200.1%20%26%200.2%0A%5Cend%7Bmatrix%7D%0A%5Cright%5D%0A" alt="
\hat y = \left[
\begin{matrix}
0.2 &amp; 0.1 &amp; 0.3 &amp; 0.4 \\
0.4 &amp; 0.2 &amp; 0.2 &amp; 0.2 \\
0.1 &amp; 0.5 &amp; 0.1 &amp; 0.2
\end{matrix}
\right]
" />

and

<img src="https://tex.s2cms.ru/svg/%0Ay%20%3D%20%5Cleft%5B%0A%5Cbegin%7Bmatrix%7D%0A4%20%5C%5C%0A1%20%5C%5C%0A2%0A%5Cend%7Bmatrix%7D%0A%5Cright%5D%0A" alt="
y = \left[
\begin{matrix}
4 \\
1 \\
2
\end{matrix}
\right]
" />

* In order to make such a comparison we need to first transform our vector of target labels <img src="https://tex.s2cms.ru/svg/%20y_i" alt=" y_i" /> such that its dimensions are consistent with the output matrix <img src="https://tex.s2cms.ru/svg/%20%5Chat%20y_%7Bik%7D%20" alt=" \hat y_{ik} " />
* For this purpose we use **one-hot encoding** which converts each element of the <img src="https://tex.s2cms.ru/svg/%20N%20%5Ctimes%201%20" alt=" N \times 1 " /> vector of target labels <img src="https://tex.s2cms.ru/svg/%20y_i%20" alt=" y_i " /> into an <img src="https://tex.s2cms.ru/svg/%20N%20%5Ctimes%20T%20" alt=" N \times T " /> matrix <img src="https://tex.s2cms.ru/svg/%20y_%7Bik%7D%20" alt=" y_{ik} " /> where in each row, the element corresponding to the correct label <img src="https://tex.s2cms.ru/svg/%20c%20" alt=" c " /> is assigned a one and the remaining elements are zero
* This is most easily illustrated with an example: the output vector from above becomes

<img src="https://tex.s2cms.ru/svg/%0Ay%20%3D%20%5Cleft%5B%0A%5Cbegin%7Bmatrix%7D%0A0%20%26%200%20%26%200%20%26%201%20%5C%5C%0A1%20%26%200%20%26%200%20%26%200%20%5C%5C%0A0%20%26%201%20%26%200%20%26%200%0A%5Cend%7Bmatrix%7D%0A%5Cright%5D%0A" alt="
y = \left[
\begin{matrix}
0 &amp; 0 &amp; 0 &amp; 1 \\
1 &amp; 0 &amp; 0 &amp; 0 \\
0 &amp; 1 &amp; 0 &amp; 0
\end{matrix}
\right]
" />

* Given this encoding we quantify the distance between our predicted (pseudo-)probabilities and the actual target labels using the multinomial cross-entropy function:

<img src="https://tex.s2cms.ru/svg/%0AJ(y%2C%5Chat%20y)%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%20%3D%201%7D%5EN%5Csum_%7Bk%20%3D%201%7D%5ET%20y_%7Bik%7D%5Clog%20%5Chat%20y_%7Bik%7D%0A" alt="
J(y,\hat y) = -\frac{1}{N} \sum_{i = 1}^N\sum_{k = 1}^T y_{ik}\log \hat y_{ik}
" />

* Since only the correct target element <img src="https://tex.s2cms.ru/svg/%20c%20" alt=" c " /> is non-zero in each row of <img src="https://tex.s2cms.ru/svg/%20y%20" alt=" y " />, the inner summation collapses and we have

<img src="https://tex.s2cms.ru/svg/%0AJ(y%2C%5Chat%20y)%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%20%3D%201%7D%5EN%20y_%7Bic%7D%5Clog%20%5Chat%20y_%7Bic%7D%0A" alt="
J(y,\hat y) = -\frac{1}{N} \sum_{i = 1}^N y_{ic}\log \hat y_{ic}
" />

* Note that for consistency throughout we have used the variable <img src="https://tex.s2cms.ru/svg/%20i%20" alt=" i " /> to index the <img src="https://tex.s2cms.ru/svg/%20N%20" alt=" N " /> samples and the variable <img src="https://tex.s2cms.ru/svg/%20k%20" alt=" k " /> to index the <img src="https://tex.s2cms.ru/svg/%20T%20" alt=" T " /> targets

##### Exercise: Show that the multinomial cross-entropy function is equivalent to the binomial one for the case <img src="https://tex.s2cms.ru/svg/%20T%20%3D%202%20" alt=" T = 2 " />.

* To summarize, we can write our feed-forward equations for the multinomial case as

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0Az%20%26%3D%20Xw%20%2B%20b%20%5C%5C%0A%5Chat%20y%20%26%3D%20%5Cmathrm%7Bsoftmax%7D(z)%20%5C%5C%0AJ%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%5Cmathbf%7B1%7D%5ET%20y%5Codot%20%5Clog%20%5Chat%20y%20%5Cmathbf%7B1%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
z &amp;= Xw + b \\
\hat y &amp;= \mathrm{softmax}(z) \\
J &amp;= -\frac{1}{N}\mathbf{1}^T y\odot \log \hat y \mathbf{1}
\end{aligned}
" />

These equations can be translated into python form as

#### Feed-forward

`z = np.dot(X, w) + b`

`y_hat = np.exp(z)/np.exp(z).sum(axis=1)`

`loss = -xlogy(y, y_hat).sum()/n_samples`

## Weekly tasks

This week we are going to repeat the exercise from week 2 of the course where we developed a "dumb" binomial classifier by writing a "dumb" multinomial classifier. (We will go through how to train the model next week.)

##### `base.py`
* In base.py fill in the function definition for `one_hot(y)` that takes a vector of target labels `y` and converts it into an <img src="https://tex.s2cms.ru/svg/%20N%20%5Ctimes%20T%20" alt=" N \times T " /> array where <img src="https://tex.s2cms.ru/svg/%20N%20" alt=" N " /> is the number of samples and <img src="https://tex.s2cms.ru/svg/%20T%20" alt=" T " /> is the number of unique labels in the list
* Fill in the definition for `log_loss(y_true, y_pred)`

#### `LogisticMultinomialClassifier`
* Start a new script `logmult.py` and create a new object `LogisticClassifierMultinomial`
* Write an internal (private) method `_initialize_weights(X)` to randomly initialize the model weights and biases given the dimension of the input array `X` – store the results as attributes, e.g. `self.w` and `self.b`. Think carefully about the dimensions of the weights and biases given the dimensions of the input data `X`
* Given the randomly initialized set of weights and biases write a method `_forward_pass(X)` to calculate the activations `z` for each input sample
* Finally, pass the activations `z` through the `softmax(z)` activation function (which you may choose to write as a separate method) and round the results to provide predictions using a `predict(X)` method – it may be best to set up `predict(X)` such that it calls `_forward_pass(X)` internally

#### `ClassifierMixin`
* Depending on your earlier implementation, you may need to update your `ClassifierMixin` object to handle multinomial datasets

Once you have completed the above tasks generate some multinomial datasets (i.e. n_targets > 2) using `DataGenerator` and `DataContainer` objects and then use your `LogisticClassifierMultinomial` object make predictions for each sample