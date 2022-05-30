---
theme: gaia
paginate: true
---

<!-- _footer: Institute for Finance and Statistics, University of Bonn -->
<!-- paginate: false -->

## Neural Networks

Topics in Econometrics and Statistics, Summer 2020

<br/>

#### Tim Mensinger

---

<!-- paginate: true -->

## Index

1. Introduction and ML Lingo
2. Opening the Black-Box
3. Training the Network
4. Application (if there is time)
5. Code (there won't be time)

---

<!-- paginate: false -->
<!-- _class: lead -->

# Introduction and ML Lingo

---

<!-- paginate: true -->

### A Neural Network


---
### Lingo

* $x$
    inputs $\equiv$ features $\equiv$ activations

* $y$
    output $\equiv$ labels $\equiv$ targets

* $(x_i, y_i)$
    observation $\equiv$ example $\equiv$ instance

---
### Data Examples

* Regression
    - $y \in \mathbb{R} \implies K = 1 \implies f : \mathbb{R}^P \to \mathbb{R}$

* Classification

    * $\tilde{y} \in \{1, \dots, K\}$ define $y := \text{one\_hot}(\tilde{y}) =
    e_{\tilde{y}}$

    * $y \in \mathbb{R}^K \implies f : \mathbb{R}^P \to \mathbb{R}^K$

    * Running Example: *Handwritten digit recognition*

---
### Example: Handwritten digit recognition

![width:550px](figures/handwritten_digit_example.png)

---

<!-- paginate: false -->
<!-- _class: lead -->

# Opening the Black-Box

---
### Architecture

* input $\overset{g_0}{\longrightarrow}$ hidden $\overset{g_1}{\longrightarrow}$ output

* $x \overset{g_0}{\longrightarrow} z \overset{g_1}{\longrightarrow} y$

* Running example: 1 hidden layer of size $H$

*
    * $g_0 : \mathbb{R}^P \to \mathbb{R}^H$
    * $g_1 : \mathbb{R}^H \to \mathbb{R}^K$
    * $f = g_1 \circ g_0$

---
### Architecture (Graph)


---
### It's just linear algebra


---
### Predict? - Regression

Given a feature-vector, how do we predict the target using $f$?

---
### Predict? - Classification

Given an image, how do we predict the label using $f$?


---
### Remark: Activation Functions


* sigmoid:

    - $\sigma(z) = 1 / (1 + \exp(-z))$

* ReLU:

    - $\sigma(z) = \max(0, z)$

    - industry standard for most tasks


---

<!-- paginate: false -->
<!-- _class: lead -->

# Training the Network

---

### Parameters (Layers)

- 1 hidden layer example ($\mathbb{R}^P \overset{g_0}{\longrightarrow} \mathbb{R}^H \overset{g_1}{\longrightarrow} \mathbb{R}^K$)

    * Parameters "$\overset{g_0}{\longrightarrow}$": $W_0 \in \mathbb{R}^{H \times P}, b_0 \in \mathbb{R}^H$

        $\implies$ #parameters $= H \times P + H$

    * Parameters "$\overset{g_1}{\longrightarrow}$": $W_1 \in \mathbb{R}^{K \times H}, b_1 \in \mathbb{R}^K$

        $\implies$ #parameters $= K \times H + K$

    * #total parameters $=: M$


---
### Parameters (Total)

* Stack them!

    $$ \theta = \left(\text{vec}(W_0), b_0, \text{vec}(W_1), b_1\right) \in \mathbb{R}^M$$


* Model

    $$ f (\cdot, \theta) : \mathbb{R}^P \to \mathbb{R}^K $$

* Task: Choose some $\theta^\ast$


---
### What do we usually do?

* Data $\{(x_i, y_i) : i = 1, \dots, n\}$

* Loss function $L : \mathbb{R}^M \to \mathbb{R}$

    * $L(\theta) := \sum_{i=1}^n \ell(f(x_i, \theta), y_i)$

    <!--
    * *mean square*: $\ell(f, y) = ||f - y||_2^2$
    * *cross entropy*: $\ell(f, y) = - y^\top \log{f}$
    -->

* Minimize!

    $$ \hat{\theta} = \theta_{min} := \underset{\theta \in \mathbb{R}^M}{\text{argmin}} \, L(\theta) $$

---
### Good News

* Activation function $\sigma$ is differentiable a.e.

* Individual loss $\ell$ is differentiable a.e.

    * $\implies L$ is differentiable a.e.


* So what about

    $$\nabla L (\hat{\theta}) \overset{!}{=} 0$$


---
### Bad News

* $L$ is non-convex (in general)
    (composition of convex functions is non-convex)

    * There are multiple local minima

    * $\theta_{min}$ may result in overfitting


* Happy with a *small enough* local minimum

    * How do we get there?


---
### Minimizing a Non-Convex Function

* General idea: Iterative Algorithm

    * Choose $\theta_0$, then apply some updating step
      $$ \theta_{k+1} \leftarrow O(L, \theta_k)$$
      until stopping criterium is reached.

    * Example: *Steepest descent*

---
### Steepest Descent
In iteration $k$, how do we choose the next point?
* *Idea*: Look for point that locally reduces the value of $L$ by the greatest amount
* Find direction $u$, with $||u||=1$, such that $L(\theta_k + u) < L(\theta_k)$
* *Taylor*:
    * $L(\theta_k + u) = L(\theta_k) + \nabla L(\theta_k)^\top u$
    * $\longrightarrow \underset{u \in \mathbb{R}^M}{\text{minimize}}$

---
### Solve the Minimization Problem
* $\min \nabla L(\theta_k)^\top u$
* $= \min ||u|| \cdot ||\nabla L(\theta_k)|| \cos(\alpha)$
* $= \min ||\nabla L(\theta_k) || \cos(\alpha)$
* minimal if $\cos(\alpha) = -1$
* $\iff \alpha = \pi$
*
* Rule:
    $\theta_{k+1} \leftarrow \theta_k - \eta \nabla L(\theta_k)$



---
### Backpropagation


* Clever application of chain-rule to get the gradient

* Made neural networks feasible + easy to teach

* But, you don't need it!

    * Use Automatic differentiation!!!

*   ```python
    from jax import grad

    def func(x):
        return very_complicated_functional_form(x)

    grad(func)(x)
    ```

---
### Neural Networks: The Complete How-To

1) Choose architecture

    * Hidden layers (size, number)

    * Activation functions (hidden, output)

2) Choose loss function

3) Choose optimizer and start parameter

4) Run optimizer and hope for the best (get $\theta^\ast$)

* Final model: $f^\ast = f(\cdot, \theta^\ast)$ !!!

---

<!-- paginate: false -->
<!-- _class: lead -->

# Application & Code

There is no time ... but you can look at it on your own.

---

### References

<style scoped>
ul li {
    font-size: 22px;
}
</style>

- Hastie, Tibshirani, & Friedman (2009). The elements of statistical learning: data mining, inference, and prediction.

- Sanderson (2017). Neural Networks. (3Blue1Brown).

- Nielsen (2019). Neural Networks and Deep Learning.

- Goodfellow, Bengio, Courville (2016). Deep Learning.

![bg width:400px right:40%](figures/qr_code_repo.png)

![](figures/GitHub-Mark-120px-plus.png)

##### Repo: `github.com/timmens/neural-net`
