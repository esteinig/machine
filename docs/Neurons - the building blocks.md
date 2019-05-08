# Neurons - the building blocks

The fundamental building blocks of neural networks are 
neurons: logistic units. The goal of logistic regression is 
to predict the result of a random experiment with a binary 
outcome (e.g. TRUE/FALSE, PASS/FAIL, 0/1), i.e. a Bernoulli 
trial that has probability $$p$$ of success. Because we are 
trying to predict a probability $$p$$, which lies in the 
range $$[0,1]$$, linear regression is not an appropriate 
model: for $$x\in(-\infty,\infty)$$ linear regression models 
for $$p$$ would predict values in the interval $$(-\infty, 
\infty)$$. Therefore, rather than modelling the probability 
$$p$$ directly with a linear regression model, we instead model the log odds. In logistic regression we are modelling the log odds with a straight line (or plane/hyperplane). If we let $$\hat y = P(y=1|x)$$, i.e. the probability that a random variable $$y$$ is equal to $$1$$ given the data $$x$$

$$\log \left(\frac{\hat y}{1 - \hat y}\right) = Xw + b \equiv z$$.

Rearranging this expression for the probability of success 
$$\hat y$$ we get

$$\hat y = \sigma (z)= \frac{1}{1+\mathrm{e}^?{-z}}$$

Here we have introduced $$\sigma (z)$$: the sigmoid or logistic 
function, $$\sigma : (-\infty, \infty)\rightarrow (0,1)$$
The sigmoid function converts an output defined over the entire real line $$\mathbb{R}$$ and maps it onto the interval $$(0,1)$$, i.e. it converts the result into a probability







The output $$\hat y$$ is a probability, we are interested in a 
prediction The predicted value is obtained via

$$ \hat Y = \mathrm{round}(\hat y) = \mathrm{round}(P(Y=1|x)) $$

Therefore, if $$\hat y \geq 0.5 $$ we predict the output has the label $$1$$
Alternatively, if $$\hat y < 0.5 $$ we predict the probability has the 
label $$0$$
Using the numpy functions we can calculate the odds and make the predictions of all samples at once
Rewrite these results in component form

$$ z_i = \sum_{j = 1}^D x_{ij} w_j + b $$

For example, if we had a dataset with 3 samples and 2 

features:
X=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )]
the activations z_i for each of the three samples are given 

by
[¦(z_1@z_2@z_3 )]=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦

(w_1@w_2 )]+b=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦(w_1@w_2 

)]+[¦(b@b@b)]
where the scalar bias b has been broadcast across each of 

the samples.
[¦(z_1@z_2@z_3 )]=[¦(x_11 w_1+x_12 w_2+b@x_21 w_1+x_22 

w_2+b@x_31 w_1+x_32 w_2+b)].

## A geometric perspective
If we look at the threshold for predicting the target label 

of a sample $$x$$, we could cut out the middle man (passing the 

activation $$z$$ through the sigmoid function $$\sigma$$) and simply use the activation $$z$$ directly:

$$ \hat y \gtrless 0.5 \iff z \gtrless 0 $$

where, $$0=x^T w+b $$ (here $$x$$ is a particular sample taken from 
the larger dataset $$X$$)

Therefore, the determining factor becomes

$$x^T w+b=0$$

This is the equation of a line (2D) or a plane (3D) or 
hyperplane (>3D)
In logistic regression we are using lines and planes to 
separate our collection of points in hyperspace

## Weekly tasks:
* Start two new scripts: “base” and “logbin” and within 
each develop two separate objects: `LogisticClassifierBinary` 
(in logbin) and a mix-in object, `ClassifierMixin` (in base)
`LogisticClassifierBinary` should take an $$N\times D$$ dataset `X` and 
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
function $$\sigma (z)$$ (which you may choose to write as a separate 
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



# Upmath: Math Online Editor
### _Create web articles and&nbsp;blog posts with&nbsp;equations and&nbsp;diagrams_

Upmath extremely simplifies this task by using Markdown and LaTeX. It converts the Markdown syntax extended with LaTeX equations support into HTML code you can publish anywhere on the web.

![Paper written in LaTeX](/i/latex.jpg)

# New equation

$$\log  $$

## Markdown

Definition from [Wikipedia](https://en.wikipedia.org/wiki/Markdown):

> Markdown is a lightweight markup language with plain text formatting syntax designed so that it can be converted to HTML and many other formats using a tool by the same name. Markdown is often used to format readme files, for writing messages in online discussion forums, and to create rich text using a plain text editor.

The main idea of Markdown is to use a simple plain text markup. It's ~~hard~~ easy to __make__ **bold** _or_ *italic* text. Simple equations can be formatted with subscripts and superscripts: *E*~0~=*mc*^2^. I have added the LaTeX support: $$E_0=mc^2$$.

Among Markdown features are:

* images (see above);
* links: [service main page](/ "link title");
* code: `untouched equation source is *E*~0~=*mc*^2^`;
* unordered lists--when a line starts with `+`, `-`, or `*`;
  1. sub-lists
  1. and ordered lists too;
* direct use <nobr>of HTML</nobr>&ndash;for <span style="color: red">anything else</span>. 

Also the editor supports typographic replacements: (c) (r) (tm) (p) +- !!!!!! ???? ,,  -- ---

## LaTeX

The editor converts LaTeX equations in double-dollars `$$`: $$ax^2+bx+c=0$$. All equations are rendered as block equations. If you need inline ones, you can add the prefix `\inline`: $$\inline p={1\over q}$$. But it is a good practice to place big equations on separate lines:

$$x_{1,2} = {-b\pm\sqrt{b^2 - 4ac} \over 2a}.$$

In this case the LaTeX syntax will be highlighted in the source code. You can even add equation numbers (unfortunately there is no automatic numbering and refs support):

$$|\vec{A}|=\sqrt{A_x^2 + A_y^2 + A_z^2}.$$(1)

It is possible to write Cyrillic symbols in `\text` command: $$Q_\text{?????????}>0$$.

One can use matrices:

$$T^{\mu\nu}=\begin{pmatrix}
\varepsilon&0&0&0\\
0&\varepsilon/3&0&0\\
0&0&\varepsilon/3&0\\
0&0&0&\varepsilon/3
\end{pmatrix},$$

integrals:

$$P_\omega={n_\omega\over 2}\hbar\omega\,{1+R\over 1-v^2}\int\limits_{-1}^{1}dx\,(x-v)|x-v|,$$

cool tikz-pictures:

$$\usetikzlibrary{decorations.pathmorphing}
\begin{tikzpicture}[line width=0.2mm,scale=1.0545]\small
\tikzset{>=stealth}
\tikzset{snake it/.style={->,semithick,
decoration={snake,amplitude=.3mm,segment length=2.5mm,post length=0.9mm},decorate}}
\def\h{3}
\def\d{0.2}
\def\ww{1.4}
\def\w{1+\ww}
\def\p{1.5}
\def\r{0.7}
\coordinate[label=below:$A_1$] (A1) at (\ww,\p);
\coordinate[label=above:$B_1$] (B1) at (\ww,\p+\h);
\coordinate[label=below:$A_2$] (A2) at (\w,\p);
\coordinate[label=above:$B_2$] (B2) at (\w,\p+\h);
\coordinate[label=left:$C$] (C1) at (0,0);
\coordinate[label=left:$D$] (D) at (0,\h);
\draw[fill=blue!14](A2)--(B2)-- ++(\d,0)-- ++(0,-\h)--cycle;
\draw[gray,thin](C1)-- +(\w+\d,0);
\draw[dashed,gray,fill=blue!5](A1)-- (B1)-- ++(\d,0)-- ++(0,-\h)-- cycle;
\draw[dashed,line width=0.14mm](A1)--(C1)--(D)--(B1);
\draw[snake it](C1)--(A2) node[pos=0.6,below] {$c\Delta t$};
\draw[->,semithick](\ww,\p+0.44*\h)-- +(\w-\ww,0) node[pos=0.6,above] {$v\Delta t$};
\draw[snake it](D)--(B2);
\draw[thin](\r,0) arc (0:atan2(\p,\w):\r) node[midway,right,yshift=0.06cm] {$\theta$};
\draw[opacity=0](-0.40,-0.14)-- ++(0,5.06);
\end{tikzpicture}$$

plots:

$$\begin{tikzpicture}[scale=1.0544]\small
\begin{axis}[axis line style=gray,
	samples=120,
	width=9.0cm,height=6.4cm,
	xmin=-1.5, xmax=1.5,
	ymin=0, ymax=1.8,
	restrict y to domain=-0.2:2,
	ytick={1},
	xtick={-1,1},
	axis equal,
	axis x line=center,
	axis y line=center,
	xlabel=$x$,ylabel=$y$]
\addplot[red,domain=-2:1,semithick]{exp(x)};
\addplot[black]{x+1};
\addplot[] coordinates {(1,1.5)} node{$y=x+1$};
\addplot[red] coordinates {(-1,0.6)} node{$y=e^x$};
\path (axis cs:0,0) node [anchor=north west,yshift=-0.07cm] {0};
\end{axis}
\end{tikzpicture}$$

and [the rest of LaTeX features](https://en.wikibooks.org/wiki/LaTeX/Mathematics).

## About Upmath

It works in browsers, except equations rendered [on the server](//tex.s2cms.com/). The editor stores your text in the browser to prevent the loss of your work in case of software or hardware failures.

I have designed and developed this lightweight editor and the service for converting LaTeX equations into svg-pictures to make publishing math texts on the web easy. I consider client-side rendering, the rival technique implemented in [MathJax](https://www.mathjax.org/), to be too limited and resource-consuming, especially on mobile devices.

The source code is [published on Github](https://github.com/parpalak/upmath.me) under MIT license.

***

Now you can erase this instruction and start writing your own scientific post. If you want to see the instruction again, open the editor in a private tab, in a different browser or download and clear your post and refresh the page.

Have a nice day :)

[Roman Parpalak](https://written.ru/), web developer and UX expert.
 
# Neurons - the building blocks

The fundamental building blocks of neural networks are 
neurons: logistic units. The goal of logistic regression is 
to predict the result of a random experiment with a binary 
outcome (e.g. TRUE/FALSE, PASS/FAIL, 0/1), i.e. a Bernoulli 
trial that has probability $$p$$ of success. Because we are 
trying to predict a probability $$p$$, which lies in the 
range $$[0,1]$$, linear regression is not an appropriate 
model: for $$x\in(-\infty,\infty)$$ linear regression models 
for $$p$$ would predict values in the interval $$(-\infty, 
\infty)$$. Therefore, rather than modelling the probability 
$$p$$ directly with a linear regression model, we instead model the log odds. In logistic regression we are modelling the log odds with a straight line (or plane/hyperplane). If we let $$\hat y = P(y=1|x)$$, i.e. the probability that a random variable $$y$$ is equal to $$1$$ given the data $$x$$

$$\log \left(\frac{\hat y}{1 - \hat y}\right) = Xw + b \equiv z$$.

Rearranging this expression for the probability of success 
$$\hat y$$ we get

$$\hat y = \sigma (z)= \frac{1}{1+\mathrm{e}^?{-z}}$$

Here we have introduced $$\sigma (z)$$: the sigmoid or logistic 
function, $$\sigma : (-\infty, \infty)\rightarrow (0,1)$$
The sigmoid function converts an output defined over the entire real line $$\mathbb{R}$$ and maps it onto the interval $$(0,1)$$, i.e. it converts the result into a probability







The output $$\hat y$$ is a probability, we are interested in a 
prediction The predicted value is obtained via

$$ \hat Y = \mathrm{round}(\hat y) = \mathrm{round}(P(Y=1|x)) $$

Therefore, if $$\hat y \geq 0.5 $$ we predict the output has the label $$1$$
Alternatively, if $$\hat y < 0.5 $$ we predict the probability has the 
label $$0$$
Using the numpy functions we can calculate the odds and make the predictions of all samples at once
Rewrite these results in component form

$$ z_i = \sum_{j = 1}^D x_{ij} w_j + b $$

For example, if we had a dataset with 3 samples and 2 

features:
X=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )]
the activations z_i for each of the three samples are given 

by
[¦(z_1@z_2@z_3 )]=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦

(w_1@w_2 )]+b=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦(w_1@w_2 

)]+[¦(b@b@b)]
where the scalar bias b has been broadcast across each of 

the samples.
[¦(z_1@z_2@z_3 )]=[¦(x_11 w_1+x_12 w_2+b@x_21 w_1+x_22 

w_2+b@x_31 w_1+x_32 w_2+b)].

## A geometric perspective
If we look at the threshold for predicting the target label 

of a sample $$x$$, we could cut out the middle man (passing the 

activation $$z$$ through the sigmoid function $$\sigma$$) and simply use the activation $$z$$ directly:

$$ \hat y \gtrless 0.5 \iff z \gtrless 0 $$

where, $$0=x^T w+b $$ (here $$x$$ is a particular sample taken from 
the larger dataset $$X$$)

Therefore, the determining factor becomes

$$x^T w+b=0$$

This is the equation of a line (2D) or a plane (3D) or 
hyperplane (>3D)
In logistic regression we are using lines and planes to 
separate our collection of points in hyperspace

## Weekly tasks:
* Start two new scripts: “base” and “logbin” and within 
each develop two separate objects: `LogisticClassifierBinary` 
(in logbin) and a mix-in object, `ClassifierMixin` (in base)
`LogisticClassifierBinary` should take an $$N\times D$$ dataset `X` and 
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
function $$\sigma (z)$$ (which you may choose to write as a separate 
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



# Upmath: Math Online Editor
### _Create web articles and&nbsp;blog posts with&nbsp;equations and&nbsp;diagrams_

Upmath extremely simplifies this task by using Markdown and LaTeX. It converts the Markdown syntax extended with LaTeX equations support into HTML code you can publish anywhere on the web.

![Paper written in LaTeX](/i/latex.jpg)

# New equation

$$\log  $$

## Markdown

Definition from [Wikipedia](https://en.wikipedia.org/wiki/Markdown):

> Markdown is a lightweight markup language with plain text formatting syntax designed so that it can be converted to HTML and many other formats using a tool by the same name. Markdown is often used to format readme files, for writing messages in online discussion forums, and to create rich text using a plain text editor.

The main idea of Markdown is to use a simple plain text markup. It's ~~hard~~ easy to __make__ **bold** _or_ *italic* text. Simple equations can be formatted with subscripts and superscripts: *E*~0~=*mc*^2^. I have added the LaTeX support: $$E_0=mc^2$$.

Among Markdown features are:

* images (see above);
* links: [service main page](/ "link title");
* code: `untouched equation source is *E*~0~=*mc*^2^`;
* unordered lists--when a line starts with `+`, `-`, or `*`;
  1. sub-lists
  1. and ordered lists too;
* direct use <nobr>of HTML</nobr>&ndash;for <span style="color: red">anything else</span>. 

Also the editor supports typographic replacements: (c) (r) (tm) (p) +- !!!!!! ???? ,,  -- ---

## LaTeX

The editor converts LaTeX equations in double-dollars `$$`: $$ax^2+bx+c=0$$. All equations are rendered as block equations. If you need inline ones, you can add the prefix `\inline`: $$\inline p={1\over q}$$. But it is a good practice to place big equations on separate lines:

$$x_{1,2} = {-b\pm\sqrt{b^2 - 4ac} \over 2a}.$$

In this case the LaTeX syntax will be highlighted in the source code. You can even add equation numbers (unfortunately there is no automatic numbering and refs support):

$$|\vec{A}|=\sqrt{A_x^2 + A_y^2 + A_z^2}.$$(1)

It is possible to write Cyrillic symbols in `\text` command: $$Q_\text{?????????}>0$$.

One can use matrices:

$$T^{\mu\nu}=\begin{pmatrix}
\varepsilon&0&0&0\\
0&\varepsilon/3&0&0\\
0&0&\varepsilon/3&0\\
0&0&0&\varepsilon/3
\end{pmatrix},$$

integrals:

$$P_\omega={n_\omega\over 2}\hbar\omega\,{1+R\over 1-v^2}\int\limits_{-1}^{1}dx\,(x-v)|x-v|,$$

cool tikz-pictures:

$$\usetikzlibrary{decorations.pathmorphing}
\begin{tikzpicture}[line width=0.2mm,scale=1.0545]\small
\tikzset{>=stealth}
\tikzset{snake it/.style={->,semithick,
decoration={snake,amplitude=.3mm,segment length=2.5mm,post length=0.9mm},decorate}}
\def\h{3}
\def\d{0.2}
\def\ww{1.4}
\def\w{1+\ww}
\def\p{1.5}
\def\r{0.7}
\coordinate[label=below:$A_1$] (A1) at (\ww,\p);
\coordinate[label=above:$B_1$] (B1) at (\ww,\p+\h);
\coordinate[label=below:$A_2$] (A2) at (\w,\p);
\coordinate[label=above:$B_2$] (B2) at (\w,\p+\h);
\coordinate[label=left:$C$] (C1) at (0,0);
\coordinate[label=left:$D$] (D) at (0,\h);
\draw[fill=blue!14](A2)--(B2)-- ++(\d,0)-- ++(0,-\h)--cycle;
\draw[gray,thin](C1)-- +(\w+\d,0);
\draw[dashed,gray,fill=blue!5](A1)-- (B1)-- ++(\d,0)-- ++(0,-\h)-- cycle;
\draw[dashed,line width=0.14mm](A1)--(C1)--(D)--(B1);
\draw[snake it](C1)--(A2) node[pos=0.6,below] {$c\Delta t$};
\draw[->,semithick](\ww,\p+0.44*\h)-- +(\w-\ww,0) node[pos=0.6,above] {$v\Delta t$};
\draw[snake it](D)--(B2);
\draw[thin](\r,0) arc (0:atan2(\p,\w):\r) node[midway,right,yshift=0.06cm] {$\theta$};
\draw[opacity=0](-0.40,-0.14)-- ++(0,5.06);
\end{tikzpicture}$$

plots:

$$\begin{tikzpicture}[scale=1.0544]\small
\begin{axis}[axis line style=gray,
	samples=120,
	width=9.0cm,height=6.4cm,
	xmin=-1.5, xmax=1.5,
	ymin=0, ymax=1.8,
	restrict y to domain=-0.2:2,
	ytick={1},
	xtick={-1,1},
	axis equal,
	axis x line=center,
	axis y line=center,
	xlabel=$x$,ylabel=$y$]
\addplot[red,domain=-2:1,semithick]{exp(x)};
\addplot[black]{x+1};
\addplot[] coordinates {(1,1.5)} node{$y=x+1$};
\addplot[red] coordinates {(-1,0.6)} node{$y=e^x$};
\path (axis cs:0,0) node [anchor=north west,yshift=-0.07cm] {0};
\end{axis}
\end{tikzpicture}$$

and [the rest of LaTeX features](https://en.wikibooks.org/wiki/LaTeX/Mathematics).

## About Upmath

It works in browsers, except equations rendered [on the server](//tex.s2cms.com/). The editor stores your text in the browser to prevent the loss of your work in case of software or hardware failures.

I have designed and developed this lightweight editor and the service for converting LaTeX equations into svg-pictures to make publishing math texts on the web easy. I consider client-side rendering, the rival technique implemented in [MathJax](https://www.mathjax.org/), to be too limited and resource-consuming, especially on mobile devices.

The source code is [published on Github](https://github.com/parpalak/upmath.me) under MIT license.

***

Now you can erase this instruction and start writing your own scientific post. If you want to see the instruction again, open the editor in a private tab, in a different browser or download and clear your post and refresh the page.

Have a nice day :)

[Roman Parpalak](https://written.ru/), web developer and UX expert.

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

For example, if we had a dataset with 3 samples and 2 

features:
X=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )]
the activations z_i for each of the three samples are given 

by
[¦(z_1@z_2@z_3 )]=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦

(w_1@w_2 )]+b=[¦(x_11&x_12@x_21&x_22@x_31&x_32 )][¦(w_1@w_2 

)]+[¦(b@b@b)]
where the scalar bias b has been broadcast across each of 

the samples.
[¦(z_1@z_2@z_3 )]=[¦(x_11 w_1+x_12 w_2+b@x_21 w_1+x_22 

w_2+b@x_31 w_1+x_32 w_2+b)].

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