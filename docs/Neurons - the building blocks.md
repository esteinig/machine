# Neurons - the building blocks

The fundamental building blocks of neural networks are 
neurons: logistic units. The goal of logistic regression is 
to predict the result of a random experiment with a binary 
outcome (e.g. TRUE/FALSE, PASS/FAIL, 0/1), i.e. a Bernoulli 
trial that has probability <img src="https://tex.s2cms.ru/svg/p" alt="p" /> of success. Because we are 
trying to predict a probability <img src="https://tex.s2cms.ru/svg/p" alt="p" />, which lies in the 
range <img src="https://tex.s2cms.ru/svg/%5B0%2C1%5D" alt="[0,1]" />, linear regression is not an appropriate 
model: for <img src="https://tex.s2cms.ru/svg/x%5Cin(-%5Cinfty%2C%5Cinfty)" alt="x\in(-\infty,\infty)" /> linear regression models 
for <img src="https://tex.s2cms.ru/svg/p" alt="p" /> would predict values in the interval <img src="https://tex.s2cms.ru/svg/(-%5Cinfty%2C%20%0A%5Cinfty)" alt="(-\infty, 
\infty)" />. Therefore, rather than modelling the probability 
<img src="https://tex.s2cms.ru/svg/p" alt="p" /> directly with a linear regression model, we instead model the log odds. In logistic regression we are modelling the log odds with a straight line (or plane/hyperplane). If we let <img src="https://tex.s2cms.ru/svg/%5Chat%20y%20%3D%20P(y%3D1%7Cx)" alt="\hat y = P(y=1|x)" />, i.e. the probability that a random variable <img src="https://tex.s2cms.ru/svg/y" alt="y" /> is equal to <img src="https://tex.s2cms.ru/svg/1" alt="1" /> given the data <img src="https://tex.s2cms.ru/svg/x" alt="x" />

<img src="https://tex.s2cms.ru/svg/%5Clog%20%5Cleft(%5Cfrac%7B%5Chat%20y%7D%7B1%20-%20%5Chat%20y%7D%5Cright)%20%3D%20Xw%20%2B%20b%20%5Cequiv%20z" alt="\log \left(\frac{\hat y}{1 - \hat y}\right) = Xw + b \equiv z" />.

Rearranging this expression for the probability of success 
<img src="https://tex.s2cms.ru/svg/%5Chat%20y" alt="\hat y" /> we get

<img src="https://tex.s2cms.ru/svg/%5Chat%20y%20%3D%20%5Csigma%20(z)%3D%20%5Cfrac%7B1%7D%7B1%2B%5Cmathrm%7Be%7D%5E%E2%81%A1%7B-z%7D%7D" alt="\hat y = \sigma (z)= \frac{1}{1+\mathrm{e}^?{-z}}" />

Here we have introduced <img src="https://tex.s2cms.ru/svg/%5Csigma%20(z)" alt="\sigma (z)" />: the sigmoid or logistic 
function, <img src="https://tex.s2cms.ru/svg/%5Csigma%20%3A%20(-%5Cinfty%2C%20%5Cinfty)%5Crightarrow%20(0%2C1)" alt="\sigma : (-\infty, \infty)\rightarrow (0,1)" />
The sigmoid function converts an output defined over the entire real line <img src="https://tex.s2cms.ru/svg/%5Cmathbb%7BR%7D" alt="\mathbb{R}" /> and maps it onto the interval <img src="https://tex.s2cms.ru/svg/(0%2C1)" alt="(0,1)" />, i.e. it converts the result into a probability







The output <img src="https://tex.s2cms.ru/svg/%5Chat%20y" alt="\hat y" /> is a probability, we are interested in a 
prediction The predicted value is obtained via

<img src="https://tex.s2cms.ru/svg/%20%5Chat%20Y%20%3D%20%5Cmathrm%7Bround%7D(%5Chat%20y)%20%3D%20%5Cmathrm%7Bround%7D(P(Y%3D1%7Cx))%20" alt=" \hat Y = \mathrm{round}(\hat y) = \mathrm{round}(P(Y=1|x)) " />

Therefore, if <img src="https://tex.s2cms.ru/svg/%5Chat%20y%20%5Cgeq%200.5%20" alt="\hat y \geq 0.5 " /> we predict the output has the label <img src="https://tex.s2cms.ru/svg/1" alt="1" />
Alternatively, if <img src="https://tex.s2cms.ru/svg/%5Chat%20y%20%3C%200.5%20" alt="\hat y &lt; 0.5 " /> we predict the probability has the 
label <img src="https://tex.s2cms.ru/svg/0" alt="0" />
Using the numpy functions we can calculate the odds and make the predictions of all samples at once
Rewrite these results in component form

<img src="https://tex.s2cms.ru/svg/%20z_i%20%3D%20%5Csum_%7Bj%20%3D%201%7D%5ED%20x_%7Bij%7D%20w_j%20%2B%20b%20" alt=" z_i = \sum_{j = 1}^D x_{ij} w_j + b " />



## A geometric perspective
If we look at the threshold for predicting the target label 

of a sample <img src="https://tex.s2cms.ru/svg/x" alt="x" />, we could cut out the middle man (passing the 

activation <img src="https://tex.s2cms.ru/svg/z" alt="z" /> through the sigmoid function <img src="https://tex.s2cms.ru/svg/%5Csigma" alt="\sigma" />) and simply use the activation <img src="https://tex.s2cms.ru/svg/z" alt="z" /> directly:

<img src="https://tex.s2cms.ru/svg/%20%5Chat%20y%20%5Cgtrless%200.5%20%5Ciff%20z%20%5Cgtrless%200%20" alt=" \hat y \gtrless 0.5 \iff z \gtrless 0 " />

where, <img src="https://tex.s2cms.ru/svg/0%3Dx%5ET%20w%2Bb%20" alt="0=x^T w+b " /> (here <img src="https://tex.s2cms.ru/svg/x" alt="x" /> is a particular sample taken from 
the larger dataset <img src="https://tex.s2cms.ru/svg/X" alt="X" />)

Therefore, the determining factor becomes

<img src="https://tex.s2cms.ru/svg/x%5ET%20w%2Bb%3D0" alt="x^T w+b=0" />

This is the equation of a line (2D) or a plane (3D) or 
hyperplane (>3D)
In logistic regression we are using lines and planes to 
separate our collection of points in hyperspace

## Weekly tasks:
* Start two new scripts: “base” and “logbin” and within 
each develop two separate objects: `LogisticClassifierBinary` 
(in logbin) and a mix-in object, `ClassifierMixin` (in base)
`LogisticClassifierBinary` should take an <img src="https://tex.s2cms.ru/svg/N%5Ctimes%20D" alt="N\times D" /> dataset `X` and 
be able to predict a binary label for each sample

#### `LogisticClassifierBinary`

* Write an internal (private) method `_initialize_weights(X)` 
to randomly initialize the model weights and biases given 
the dimension of the input array `X` – store the results as 
attributes, e.g. `self.w` and `self.b`. Think carefully about 
the dimensions of the weights and biases given the 
dimensions of the input data `X`.

* Given the randomly initialized set of weights and biases 
write a method `_forward_pass(X)` to calculate the 
activations `z` for each input sample

* Finally, pass the activations `z` through the activation 
function <img src="https://tex.s2cms.ru/svg/%5Csigma%20(z)" alt="\sigma (z)" /> (which you may choose to write as a separate 
method) and round the results to provide predictions using 
a `predict(X)` method – it may be best to set up `predict(X)` 
such that it calls `predict(X)` internally

#### `ClassifierMixin`

* The `ClassifierMixin` object should have several methods 
that can be used to summarize the accuracy of the `LogisticClassifier`

* Write a method `score()` that returns the accuracy of the 
logistic classifier predictions when compared with the true 
labels given in `y`

* Write a method `misclassified()` that identifies the 
samples that were mislabeled according to your model

* Write a method `plot_fit()` that plots the dataset along 
with the separating boundary line/surface predicted by the 
model and indicates which points have been classified 
correctly or incorrectly

* Generate some random binary datasets (using your 
`DataGenerator` and `DataContainer` objects developed last 
week) and use your `LogisticClassifier` model to predict 
labels for each sample. 


Once you have completed these tasks you will have developed 
a “dumb” classifier that most likely does a poor job of 
discriminating between the targets. (Stop and think: For a 
binary problem, what would be a reasonable baseline 
accuracy?) The model is dumb only because we haven’t 
trained it yet. The weights and biases that we have used 
were initialized randomly so that they have no connection 
to the actual dataset we are trying make predictions on. In 
the next lecture we will learn how to tune the weights 
biases (i.e. train the model) in order to make more 
accurate predictions.