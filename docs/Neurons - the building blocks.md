# Neurons - the building blocks

The fundamental building blocks of neural networks are neurons: logistic units. The goal of logistic regression is to predict the result of a random experiment with a binary outcome (e.g. TRUE/FALSE, PASS/FAIL, 0/1), i.e. a Bernoulli trial that has probability $p$ of success. Because we are trying to predict a probability $p$, which lies in the range $[0,1]$, linear regression is not an appropriate model: for $x\in(-\infty,\infty)$ linear regression models for $p$ would predict values in the interval $(-\infty, \infty)$. Therefore, rather than modelling the probability $p$ directly with a linear regression model, we instead model the log odds. In logistic regression we are modelling the log odds with a straight line (or plane/hyperplane). If we let $\hat y = P(y=1|x)$, i.e. the probability that a random variable $y$ is equal to $1$ given the data $x$:
$\log \left(\frac{\hat y}{1 - \hat y}\right) = Xw + b \equiv z$.
Rearranging this expression for the probability of success $\hat y$ we get
y ^=s(z)=1/(1+exp??(-z)? )
Here we have introduced s(z): the sigmoid or logistic function, s:(-8,8)?(0,1)
The sigmoid function converts an output defined over the entire real line and maps it onto the interval (0,1), i.e. it converts the result into a probability







The output y ^ is a probability, we are interested in a prediction
The predicted value is obtained via
Y ^=round(y ^ )=round(P(y=1¦x))
Therefore, if y ^=0.5 we predict the output has the label 1
Alternatively, if y ^<0.5 we predict the probability has the label 0
Using the numpy functions we can calculate the odds and make the predictions of all samples at once
Rewrite these results in component form
z_i=?_(j=1)^D¦?x_ij w_j ?+b
For example, if we had a dataset with 3 samples and 2 features:
X=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )]
the activations z_i for each of the three samples are given by
[¦(z_1@z_2@z_3 )]=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦(w_1@w_2 )]+b=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦(w_1@w_2 )]+[¦(b@b@b)]
where the scalar bias b has been broadcast across each of the samples.
[¦(z_1@z_2@z_3 )]=[¦(x_11 w_1+x_12 w_2+b@x_21 w_1+x_22 w_2+b@x_31 w_1+x_32 w_2+b)].

## A geometric perspective
If we look at the threshold for predicting the target label of a sample x, we could cut out the middle man (passing the activation z through the sigmoid function s) and simply use the activation z directly:
y ^?0.5 ?z?0
where, 0=x^T w+b (here x is a particular sample taken from the larger dataset X)

Therefore, the determining factor becomes
x^T w+b=0
This is the equation of a line (2D) or a plane (3D) or hyperplane (>3D)
In logistic regression we are using lines and planes to separate our collection of points in hyperspace

## Weekly tasks:
* Start two new scripts: “base” and “logbin” and within each develop two separate objects: LogisticClassifierBinary (in logbin) and a mix-in object, ClassifierMixin (in base)
LogisticClassifierBinary should take an N×D dataset X and be able to predict a binary label for each sample
* Write an internal (private) method _initialize_weights(X) to randomly initialize the model weights and biases given the dimension of the input array X – store the results as attributes, e.g. self.w and self.b. Think carefully about the dimensions of the weights and biases given the dimensions of the input data X.
* Given the randomly initialized set of weights and biases write a method _forward_pass(X) to calculate the activations z for each input sample
* Finally, pass the activations z through the activation function s(z) (which you may choose to write as a separate method) and round the results to provide predictions using a predict(X) method – it may be best to set up predict(X) such that it calls predict(X) internally
### The ClassifierMixin object should have several methods that can be used to summarize the accuracy of the LogisticClassifier
* Write a method score() that returns the accuracy of the logistic classifier predictions when compared with the true labels given in y
* Write a method misclassified() that identifies the samples that were mislabeled according to your model
* Write a method plot_fit() that plots the dataset along with the separating boundary line/surface predicted by the model and indicates which points have been classified correctly or incorrectly
* Generate some random binary datasets (using your DataGenerator and DataContainer objects developed last week) and use your LogisticClassifier model to predict labels for each sample. 

Once you have completed these tasks you will have developed a “dumb” classifier that most likely does a poor job of discriminating between the targets. (Stop and think: For a binary problem, what would be a reasonable baseline accuracy?) The model is dumb only because we haven’t trained it yet. The weights and biases that we have used were initialized randomly so that they have no connection to the actual dataset we are trying make predictions on. In the next lecture we will learn how to tune the weights biases (i.e. train the model) in order to make more accurate predictions.

