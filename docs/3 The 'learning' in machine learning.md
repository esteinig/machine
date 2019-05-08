# The “learning” in machine learning

- Given a set of weights (<img src="https://tex.s2cms.ru/svg/w" alt="w" />) and biases (<img src="https://tex.s2cms.ru/svg/b" alt="b" />) we can easily predict the target (<img src="https://tex.s2cms.ru/svg/y" alt="y" />) given a particular dataset <img src="https://tex.s2cms.ru/svg/X" alt="X" />
- The question is, how do we find the weights and biases to provide accurate predictions?
- The model needs to be trained, i.e. it needs to learn the correct weights and biases
- This is the ‘learning’ in ‘machine learning’
- The way a machine learning model learns is by comparing the predicted outputs with known target values
- We therefore need some (quantitative) method of comparing predicted targets <img src="https://tex.s2cms.ru/svg/%5Chat%20Y" alt="\hat Y" /> with actual labels <img src="https://tex.s2cms.ru/svg/Y" alt="Y" />
- Motivated by this, we introduce a loss function (similar to the squared error function used in linear regression) that measures the “distance” between our predictions and the observations, i.e. a function that takes on large values when our model makes many incorrect predictions and smaller ones when our model is more accurate
- For binary classification problems, the most common choice for the loss function is the binary cross-entropy function

<img src="https://tex.s2cms.ru/svg/%20J%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20y_i%20%5Clog%20%5Chat%20y_i%20%2B%20(1%20-%20y_i)%20%5Clog%20(1%20-%20%5Chat%20y_i)%20" alt=" J = -\frac{1}{N} \sum_{i=1}^N y_i \log \hat y_i + (1 - y_i) \log (1 - \hat y_i) " />

- Here, <img src="https://tex.s2cms.ru/svg/y_i%5Cin%5C%7B0%2C1%5C%7D" alt="y_i\in\{0,1\}" /> is the actual or true label of the i-th sample
- <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i%5Cin%20(0%2C1)" alt="\hat y_i\in (0,1)" /> is the predicted probability that the target label is equal to one
- The sum is taken over all samples <img src="https://tex.s2cms.ru/svg/i%5Cin%5C%7B1%2C2%2C%E2%80%A6%2CN%5C%7D" alt="i\in\{1,2,…,N\}" />
- Note that since the actual target labels (<img src="https://tex.s2cms.ru/svg/y" alt="y" />) take either the value <img src="https://tex.s2cms.ru/svg/0" alt="0" /> or <img src="https://tex.s2cms.ru/svg/1" alt="1" />, for fixed <img src="https://tex.s2cms.ru/svg/i" alt="i" />, only one of the logarithmic terms contributes to the sum
- Our objective will be to find the weights and biases that minimize this sum
- The binary cross-entropy function is actually the negative log-likelihood for a series of <img src="https://tex.s2cms.ru/svg/N" alt="N" /> independent Bernoulli trials, each having probability <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i" alt="\hat y_i" /> of success (i.e. the probability that <img src="https://tex.s2cms.ru/svg/y_i%3D1" alt="y_i=1" />)

<img src="https://tex.s2cms.ru/svg/J%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Clog%20L" alt="J = -\frac{1}{N} \log L" />

where

<img src="https://tex.s2cms.ru/svg/%20L%20%3D%20%5Cprod_%7Bi%3D1%7D%5EN%20%5Chat%20y_i%5E%7By_i%7D%20(1%20-%20%5Chat%20y_i)%5E%7B1%20-%20y_i%7D%20" alt=" L = \prod_{i=1}^N \hat y_i^{y_i} (1 - \hat y_i)^{1 - y_i} " />
	
- If this formula appears a little convoluted, consider first the simplest case, where the probability of success <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i" alt="\hat y_i" /> is equal across all trials, i.e. <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i%3Dp" alt="\hat y_i=p" /> for all <img src="https://tex.s2cms.ru/svg/i" alt="i" />
- An example of this is tossing a (biased) coin <img src="https://tex.s2cms.ru/svg/N" alt="N" /> times where <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i%3Dp" alt="\hat y_i=p" /> might represent the probability of getting heads
- In this case, the likelihood <img src="https://tex.s2cms.ru/svg/L" alt="L" /> of observing a particular sequence of coin flips with <img src="https://tex.s2cms.ru/svg/i" alt="i" /> heads from <img src="https://tex.s2cms.ru/svg/N" alt="N" /> tosses is given by



<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0AL%20%26%3D%20P(H)P(H)P(T)%5Cldots%20P(T)P(H)%5C%5C%0A%20%26%3D%20P(H)%5Ei%20P(T)%5E%7BN-i%7D%20%5C%5C%0A%26%3D%20p%5Ei(1-p)%5E%7BN-i%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
L &amp;= P(H)P(H)P(T)\ldots P(T)P(H)\\
 &amp;= P(H)^i P(T)^{N-i} \\
&amp;= p^i(1-p)^{N-i}
\end{aligned}
" />

- Our goal is to find the value of <img src="https://tex.s2cms.ru/svg/p" alt="p" /> that maximizes the likelihood <img src="https://tex.s2cms.ru/svg/L" alt="L" />
- That is, what is the probability or bias <img src="https://tex.s2cms.ru/svg/p" alt="p" /> of observing heads that it is most likely to result in the observed sequence of flips
- To maximize <img src="https://tex.s2cms.ru/svg/L" alt="L" /> w.r.t. <img src="https://tex.s2cms.ru/svg/p" alt="p" /> we differentiate, set to zero and solve for <img src="https://tex.s2cms.ru/svg/p" alt="p" />:

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7BdL%7D%7Bdp%7D%20%26%3D%20ip%5E%7Bi-1%7D(1-p)%5E%7BN-i%7D%20-%20(N%20-%20i)p%5Ei(1-p)%5E%7BN-i-1%7D%20%5C%5C%0A%26%3D%20p%5E%7Bi-1%7D(1-p)%5E%7BN-i-1%7D%5Bi(1-p)%20-%20(N-i)p%5D%20%5C%5C%0A%26%3D%20p%5E%7Bi-1%7D(1-p)%5E%7BN-i-1%7D%5Bi%20-%20Np%5D%20%5C%5C%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{dL}{dp} &amp;= ip^{i-1}(1-p)^{N-i} - (N - i)p^i(1-p)^{N-i-1} \\
&amp;= p^{i-1}(1-p)^{N-i-1}[i(1-p) - (N-i)p] \\
&amp;= p^{i-1}(1-p)^{N-i-1}[i - Np] \\
\end{aligned}
" />

- Setting <img src="https://tex.s2cms.ru/svg/dL%2Fdp%20%3D%200" alt="dL/dp = 0" /> and solving for <img src="https://tex.s2cms.ru/svg/p" alt="p" /> gives

<img src="https://tex.s2cms.ru/svg/%20p%20%3D%20%5Cfrac%7Bi%7D%7BN%7D%20" alt=" p = \frac{i}{N} " />

- As expected, the most likely value for <img src="https://tex.s2cms.ru/svg/p" alt="p" /> (the bias of the coin) is just the proportion of observed heads
- Alternatively, since <img src="https://tex.s2cms.ru/svg/%5Clog%20%E2%81%A1x" alt="\log ?x" /> is monotonic in <img src="https://tex.s2cms.ru/svg/x" alt="x" /> we could instead maximize the log-likelihood, or equivalently, minimize the negative log-likelihood

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0AJ%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Clog%20L%20%5C%5C%0A%26%3D%20-%5Cfrac%7Bi%7D%7BN%7D%20%5Clog%20p%20-%20%5Cfrac%7BN-i%7D%7BN%7D%20%5Clog%20(1-p)%20%5C%5C%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
J &amp;= -\frac{1}{N} \log L \\
&amp;= -\frac{i}{N} \log p - \frac{N-i}{N} \log (1-p) \\
\end{aligned}
" />

- Differentiating w.r.t <img src="https://tex.s2cms.ru/svg/p" alt="p" /> and setting to zero gives

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7BdJ%7D%7Bdp%7D%20%26%3D%20-%5Cfrac%7Bi%7D%7BNp%7D%20%2B%20%5Cfrac%7BN-i%7D%7BN(1-p)%7D%20%5C%5C%0A0%26%3D%20%5Cfrac%7B-i(1-p)%20%2B%20(N-i)p%7D%7BNp(1-p)%7D%20%5C%5C%0A0%26%3D%20%5Cfrac%7B-i%20%2B%20Np%7D%7BNp(1-p)%7D%20%5C%5C%0A%26%5C%5C%0A%5CRightarrow%20p%20%26%3D%20%5Cfrac%7Bi%7D%7BN%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{dJ}{dp} &amp;= -\frac{i}{Np} + \frac{N-i}{N(1-p)} \\
0&amp;= \frac{-i(1-p) + (N-i)p}{Np(1-p)} \\
0&amp;= \frac{-i + Np}{Np(1-p)} \\
&amp;\\
\Rightarrow p &amp;= \frac{i}{N}
\end{aligned}
" />

- For the general case where each of the <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i" alt="\hat y_i" /> are distinct (i.e. unequal) the maximum likelihood problem cannot be solved analytically (unlike the coin-toss example)
- In order to optimize the loss function <img src="https://tex.s2cms.ru/svg/J" alt="J" /> we must resort to some numerical method

## Gradient descent
- At this point we remind ourselves that our goal, stated mathematically, is to optimize the loss function <img src="https://tex.s2cms.ru/svg/J" alt="J" />
- For the binary cross-entropy function we have introduced above the solution to this problem cannot be found analytically
- The workhorse of many (all?) machine learning models is gradient descent
- We remind ourselves that the loss function <img src="https://tex.s2cms.ru/svg/J" alt="J" /> is a function of the model parameters, i.e. the weights and biases: <img src="https://tex.s2cms.ru/svg/J%5Cequiv%20J(w%2Cb)" alt="J\equiv J(w,b)" />
- Consider first a 1d example with the loss function plotted on the vertical axis and our single bias <img src="https://tex.s2cms.ru/svg/b" alt="b" /> along the horizontal axis:
 
 
- We wish to find the minimum of this function with respect to our model parameters
- First let’s review the feed-forward equations used to predict the target labels and calculate the total loss:

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0Az_i%20%26%3D%20%5Csum_%7Bj%3D1%7D%5ED%20x_%7Bij%7Dw_j%20%2B%20b%20%5C%5C%0A%5Chat%20y_i%20%26%3D%20%5Csigma(z_i)%20%5C%5C%0AJ%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20y_i%20%5Clog%20%5Chat%20y_i%20%2B%20(1%20-%20y_i)%5Clog%20(1%20-%20%5Chat%20y_i)%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
z_i &amp;= \sum_{j=1}^D x_{ij}w_j + b \\
\hat y_i &amp;= \sigma(z_i) \\
J &amp;= -\frac{1}{N} \sum_{i=1}^N y_i \log \hat y_i + (1 - y_i)\log (1 - \hat y_i)
\end{aligned}
" />

- Note that here the index <img src="https://tex.s2cms.ru/svg/i%5Cin%20%5C%7B1%2C2%2C%E2%80%A6%2CN%5C%7D" alt="i\in \{1,2,…,N\}" /> counts over the <img src="https://tex.s2cms.ru/svg/N" alt="N" /> samples whilst the index <img src="https://tex.s2cms.ru/svg/j%5Cin%20%5C%7B1%2C2%2C%E2%80%A6%2CD%5C%7D" alt="j\in \{1,2,…,D\}" /> counts the number of dimensions or features
- In vector form our feed-forward equations become

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0Az%20%26%3D%20Xw%20%2B%20b%20%5C%5C%0Ay%20%26%3D%20%5Csigma(z)%20%5C%5C%0AJ%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Cmathbf%7B1%7D%5ET%20y%5Clog%20%5Chat%20y%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
z &amp;= Xw + b \\
y &amp;= \sigma(z) \\
J &amp;= -\frac{1}{N} \mathbf{1}^T y\log \hat y
\end{aligned}
" />

where <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7B1%7D" alt="\mathbf{1}" /> is a vector of all ones
- We want to determine how the cost function changes with respect to our weights and biases, i.e. we want to calculate the gradients <img src="https://tex.s2cms.ru/svg/%5Cnabla_w%20J" alt="\nabla_w J" /> and <img src="https://tex.s2cms.ru/svg/%5Cnabla_b%20J" alt="\nabla_b J" />.
- Knowing that <img src="https://tex.s2cms.ru/svg/J%3DJ(%5Chat%20y)%3DJ(%5Chat%20y(z))%3DJ(y(z(w%2Cb)))" alt="J=J(\hat y)=J(\hat y(z))=J(y(z(w,b)))" /> we can use successive applications of the chain rule to find

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_j%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_i%7D%20%5Cfrac%7B%5Cpartial%20%5Chat%20y_i%7D%7B%5Cpartial%20z_i%7D%5Cfrac%7B%5Cpartial%20z_i%7D%7B%5Cpartial%20w_j%7D%0A" alt="
\frac{\partial J}{\partial w_j} = \sum_{i=1}^N \frac{\partial J}{\partial \hat y_i} \frac{\partial \hat y_i}{\partial z_i}\frac{\partial z_i}{\partial w_j}
" />

- Let’s look at all of the individual pieces separately (see https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0Az_i%20%26%3D%20%5Csum_%7Bj%3D1%7D%5ED%20x_%7Bij%7Dw_j%20%2B%20b%20%5C%5C%0A%5CRightarrow%20%5Cfrac%7B%5Cpartial%20z_i%7D%7B%5Cpartial%20w_j%7D%20%26%3D%20x_%7Bij%7D%5C%5C%0A%5C%5C%0A%5Chat%20y_i%20%26%3D%20%5Csigma(z_i)%20%5C%5C%0A%5CRightarrow%20%5Cfrac%7B%5Cpartial%20%5Chat%20y_i%7D%7B%5Cpartial%20z_i%7D%20%26%3D%20%5Csigma'(z_i)%20%5C%5C%0A%26%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20z_i%7D%20%5Cfrac%7B1%7D%7B1%20%2B%20%5Cmathrm%7Be%7D%5E%7B-z_i%7D%20%5C%5C%0A%26%3D%20%5Cfrac%7B%5Cmathrm%7Be%7D%5E%7B-z_i%7D%7D%7B(1%20%2B%20%5Cmathrm%7Be%7D%5E%7B-z_i%7D)%5E2%7D%20%5C%5C%0A%26%3D%20%5Chat%20y_i(1%20-%20%5Chat%20y_i)%5C%5C%0A%5C%5C%0AJ%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%20y_i%20%5Clog%20%5Chat%20y_i%20%2B%20(1%20-%20y_i)%5Clog%20(1%20-%20%5Chat%20y_i)%20%5C%5C%0A%5CRightarrow%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_i%7D%20%26%3D%20-%5Cfrac%7B1%7D%7BN%7D%5Cleft(%5Cfrac%7By_i%7D%7B%5Chat%20y_i%7D%20%2B%20%5Cfrac%7B1%20-%20y_i%7D%7B1%20%20-%5Chat%20y_i%7D%5Cright)%5C%5C%0A%26%3D%20%5Cfrac%7B1%7D%7BN%7D%5Cfrac%7B%5Chat%20y_i%20-%20y_i%7D%7B%5Chat%20y_i(1%20-%20%5Chat%20y_i)%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
z_i &amp;= \sum_{j=1}^D x_{ij}w_j + b \\
\Rightarrow \frac{\partial z_i}{\partial w_j} &amp;= x_{ij}\\
\\
\hat y_i &amp;= \sigma(z_i) \\
\Rightarrow \frac{\partial \hat y_i}{\partial z_i} &amp;= \sigma'(z_i) \\
&amp;= \frac{\partial}{\partial z_i} \frac{1}{1 + \mathrm{e}^{-z_i} \\
&amp;= \frac{\mathrm{e}^{-z_i}}{(1 + \mathrm{e}^{-z_i})^2} \\
&amp;= \hat y_i(1 - \hat y_i)\\
\\
J &amp;= -\frac{1}{N}\sum_{i=1}^N y_i \log \hat y_i + (1 - y_i)\log (1 - \hat y_i) \\
\Rightarrow\frac{\partial J}{\partial \hat y_i} &amp;= -\frac{1}{N}\left(\frac{y_i}{\hat y_i} + \frac{1 - y_i}{1  -\hat y_i}\right)\\
&amp;= \frac{1}{N}\frac{\hat y_i - y_i}{\hat y_i(1 - \hat y_i)}
\end{aligned}
" />

- To simplify the second derivative <img src="https://tex.s2cms.ru/svg/%5Cpartial%20%5Chat%20y_i%2F%5Cpartial%20z_i" alt="\partial \hat y_i/\partial z_i" /> we have used the identities

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Chat%20y_i%20%26%3D%20%5Cfrac%7B1%7D%7B1%2B%5Cmathrm%7Be%7D%5E%7B-z_i%7D%7D%5C%5C%0A1-%5Chat%20y_i%26%3D%5Cfrac%7B%5Cmathrm%7Be%7D%5E%7B-z_i%7D%7D%7B1%2B%5Cmathrm%7Be%7D%5E%7B-z_i%7D%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\hat y_i &amp;= \frac{1}{1+\mathrm{e}^{-z_i}}\\
1-\hat y_i&amp;=\frac{\mathrm{e}^{-z_i}}{1+\mathrm{e}^{-z_i}}
\end{aligned}
" />

- Note that the second and third derivatives depend on the choice of the loss function <img src="https://tex.s2cms.ru/svg/J" alt="J" /> and output activation function <img src="https://tex.s2cms.ru/svg/%5Csigma%20(z)" alt="\sigma (z)" />. However, the first derivative, which in vector form becomes <img src="https://tex.s2cms.ru/svg/%5Cpartial%20z%2F%5Cpartial%20w%3DX" alt="\partial z/\partial w=X" />
remains consistent for all linear-based models
- Putting it all together gives

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20w_j%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20x_%7Bij%7D(%5Chat%20y_i%20-%20y_i)%0A" alt="
\frac{\partial J}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N x_{ij}(\hat y_i - y_i)
" />

- Similarly, the change in <img src="https://tex.s2cms.ru/svg/J" alt="J" /> with respect to the biases <img src="https://tex.s2cms.ru/svg/b" alt="b" /> is

<img src="https://tex.s2cms.ru/svg/%0A%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_i%7D%5Cfrac%7B%5Cpartial%20%5Chat%20y_i%7D%7B%5Cpartial%20z_i%7D%20%5Cfrac%7B%5Cpartial%20z_i%7D%7B%5Cpartial%20b%7D%0A" alt="
\frac{\partial J}{\partial b} = \sum_{i=1}^N \frac{\partial J}{\partial \hat y_i}\frac{\partial \hat y_i}{\partial z_i} \frac{\partial z_i}{\partial b}
" />

- The first two pieces have been calculated already, and the last piece is simply

<img src="https://tex.s2cms.ru/svg/%20%5Cfrac%7B%5Cpartial%20z_i%7D%7B%5Cpartial%20b%7D%20%3D%201%20" alt=" \frac{\partial z_i}{\partial b} = 1 " />

- This gives

<img src="https://tex.s2cms.ru/svg/%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20b%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20(%5Chat%20y_i%20-%20y_i)%20" alt=" \frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat y_i - y_i) " />


- If we treat the target vector <img src="https://tex.s2cms.ru/svg/y" alt="y" /> and the prediction vector <img src="https://tex.s2cms.ru/svg/%5Chat%20y" alt="\hat y" /> as column vectors (where each element/row represents a different sample) we can rewrite these expression as

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cnabla_w%20J%20%26%3D%20%5Cfrac%7B1%7D%7BN%7D%20X%5ET(%5Chat%20y%20-%20y)%20%5C%5C%0A%5Cnabla_b%20J%20%26%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Cmathbf%7B1%7D%5ET(%5Chat%20y%20-%20y)%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\nabla_w J &amp;= \frac{1}{N} X^T(\hat y - y) \\
\nabla_b J &amp;= \frac{1}{N} \mathbf{1}^T(\hat y - y)
\end{aligned}
" />
- If we introduce the mean error vector <img src="https://tex.s2cms.ru/svg/%5Cdelta%20%3D%20(%5Chat%20y%20-%20y)%2FN" alt="\delta = (\hat y - y)/N" /> we can rewrite these expressions as

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cnabla_w%20J%20%26%3D%20X%5ET%5Cdelta%20%5C%5C%0A%5Cnabla_b%20J%20%26%3D%20%5Cmathbf%7B1%7D%5ET%5Cdelta%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\nabla_w J &amp;= X^T\delta \\
\nabla_b J &amp;= \mathbf{1}^T\delta
\end{aligned}
" />

- Pay close attention to the form of these equations as they will pop up repeatedly throughout this section of the course
- Tracing back the origins of the mean error term <img src="https://tex.s2cms.ru/svg/%5Cdelta" alt="\delta" /> we see that a general expression for <img src="https://tex.s2cms.ru/svg/%5Cdelta" alt="\delta" /> is given by

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cdelta_i%20%26%3D%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Chat%20y_i%7D%20%5Cfrac%7B%5Cpartial%20%5Chat%20y_i%7D%7B%5Cpartial%20z_i%7D%20%5C%5C%0A%5Cdelta%20%26%3D%20%5Cnabla_%7B%5Chat%20y%7DJ%20%5Codot%20%5Csigma'(z)%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\delta_i &amp;= \frac{\partial J}{\partial \hat y_i} \frac{\partial \hat y_i}{\partial z_i} \\
\delta &amp;= \nabla_{\hat y}J \odot \sigma'(z)
\end{aligned}
" />

where <img src="https://tex.s2cms.ru/svg/%5Codot" alt="\odot" /> denotes the Hadamard or element-wise product

- Once we have calculated the gradient of the loss function with respect to the weights and biases we use gradient descent to update the model parameters:

<img src="https://tex.s2cms.ru/svg/%0A%5Cbegin%7Baligned%7D%0Aw%20%5Crightarrow%20%26%20%5C%2Cw%20-%20%5Ceta%20%5Cnabla_w%20J%20%5C%5C%0Ab%20%5Crightarrow%20%26%20%5C%2Cb%20-%20%5Ceta%20%5Cnabla_b%20J%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
w \rightarrow &amp; \,w - \eta \nabla_w J \\
b \rightarrow &amp; \,b - \eta \nabla_b J
\end{aligned}
" />

- Here, <img src="https://tex.s2cms.ru/svg/%5Ceta" alt="\eta" /> is the learning rate (a hyper-parameter)
- After updating the weights, we iterate:
1. Feed-forward – calculate the output probability <img src="https://tex.s2cms.ru/svg/%5Chat%20y_i" alt="\hat y_i" /> given the samples <img src="https://tex.s2cms.ru/svg/x_i" alt="x_i" />
2. Calculate the gradient of the loss function, <img src="https://tex.s2cms.ru/svg/%5Cnabla_w%20J" alt="\nabla_w J" /> and <img src="https://tex.s2cms.ru/svg/%5Cnabla_b%20J" alt="\nabla_b J" />
3. Update the weights and biases using the loss gradient
4. Repeat steps 1 through 3

For convenience, I have summarized the feed-forward and backpropagation equations that you will need to enter into python below

#### Feed-forward



`z = np.dot(X,w) + b`

`y_hat = 1/(1 + np.exp(-z))`

`loss = -(xlogy(y, y_hat) + xlogy(1-y, 1-y_hat)).sum()/y.shape[0]`

#### Backpropagation

`delta = (y_hat - y)/y.shape[0]`

`grad_w = np.dot(X.T, delta)`

`grad_b = np.sum(delta, axis=0)`

`w -= self.learning_rate * grad_w`

`b -= self.learning_rate * grad_b`



## Weekly tasks
This week we will update our `LogisticClassifier` object by providing it with the ability to train by adding the method `fit()`.
- First, return to the `DataContainer` object and add a method `train_test_split(frac=0.8)` that splits the dataset `(X,y)` into test and training datasets assigning some fraction `frac` to the training component
- Add a function `binary_log_loss(y_true, y_prob)` to the base.py script to compute the loss <img src="https://tex.s2cms.ru/svg/J" alt="J" /> of a particular training dataset (whether that be an individual sample, a batch, or the entire set)
- In `LogisticClassifier` write an internal (private) method `_compute_loss_grad(X, delta)` that computes the gradient of loss with respect to the weights and biases
- Write another internal method `_update_params(grad_w, grad_b)` that updates the weights and biases based on the loss gradients
- Write a `fit(X, y)` method that accepts a sample dataset `X` and target set `y` and trains the model to minimize the loss function
	
Once you have written each of the methods above try your new model out on one of your toy datasets. Make sure to split the dataset into training and test sets: fitting the model to the training set and then validating the model fit on the test set. For which datasets does the logistic classifier work best and for which does it provide a poor fit?
