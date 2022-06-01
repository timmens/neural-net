---
theme: gaia
paginate: true
---

<!-- _footer: Institute for Finance and Statistics, University of Bonn -->
<!-- paginate: false -->

## Neural Networks

Topics in Econometrics and Statistics, Summer 2022

<br/>

#### Tim Mensinger

---

<!-- paginate: true -->

## Index

1. Introduction
2. Approximation Theorems
3. Training
4. Bonus (Gradient Descent)

---

<!-- paginate: false -->
<!-- _class: lead -->

# Introduction

---

<!-- paginate: true -->

<style>
mjx-container {
    font-size: 50px;
}
</style>

### A Parametrized Function
<br/>
<br/>
<br/>

$$\begin{align}
f(\cdot, &\theta) : \mathbb{R}^P \to \mathbb{R}\\[0.5em]
&\theta \in \mathbb{R}^T
\end{align}$$

---
### A Neural Network

---
### One Hidden Layer

---
### One Hidden Layer (contd.)


---
### Multiple Hidden Layers

---
### Multiple Hidden Layers (contd.)

---
### Where are we?
<br/>

* Opened the black-box

* Neural networks are just parametrized functions

* What now?


---
<style>
mjx-container {
    font-size: 30px;
}
</style>
### Theorem (Weierstrass, 1885)

<br/>

Let $g \in C([0, 1], \mathbb{R})$, then there exists a sequence of polynomials $\{p_n\}$
that converges uniformly to $g$.

<br/>


* Can we get a similar result for neural networks?

---

<!-- paginate: false -->
<!-- _class: lead -->

# Approximation Theorems

---
<!-- paginate: true -->

### Two cases

<br/>
<br/>

1) Arbitrary **width**, fixed *depth*

2) Arbitrary *depth*, fixed **width**


---
### Theorem (Cybenko; Hornik; Pinkus)

* Let $\sigma : \mathbb{R} \to \mathbb{R}$ be any continuous function that is non-polynomial.

* Let $\mathcal{N}_P^\sigma$ represent the class of NN's with activation function $\sigma$, input dimension $P$ and **one** hidden layer with arbitrary number of neurons.

<br/>

* **Then:** $\mathcal{N}_P^\sigma$ is dense in $C([0, 1]^P, \mathbb{R})$.

---
###


---
### Theorem (Kidger & Lyons, 2020)

* Let $\sigma : \mathbb{R} \to \mathbb{R}$ be any non-affine cont. function s.t.  $\exists x: \sigma'(x) \neq 0$ and $\sigma'$ cont. at $x$.

* Let $\mathcal{NN}_{P, H}^\sigma$ represent the class of NN's with activation function $\sigma$, input dimension $P$ and an arbitrary number of hidden layers each with fixed width $H$.

<br/>

* **Then:** $\mathcal{NN}_{P, H}^\sigma$ is dense in $C([0,1]^P, \mathbb{R})$ **if** $H \geq P + 3$.

---
### Caveats

* Existence results

* How large do $H$ and $L$ need to be in practice?

---

<!-- paginate: false -->
<!-- _class: lead -->

# Training


---
<!-- paginate: true -->
### Parameters

* Let $f(\cdot, \theta)$ be a NN with input dimension $P$ and $L$ hidden layers of size $H$.

* $\theta = \left(\text{vec}(W_0), b_0, \text{vec}(W_1), b_1, \dots, \text{vec}(W_L), b_L, a \right)$

* $\dim(\theta) = PH + H + H^2 + H + \dots + H^2 + H + H = \mathcal{O}(PH + LH^2)$


---
### Loss Function

* Data $\{(x_i, y_i) : i = 1, \dots, n\}$
<br/>
* $\mathbb{L}(\theta) = \frac{1}{n} \sum_i (y_i - f(x_i, \theta))^2$
    * $\implies$ yields approximation of $\mathbb{E}[y_i|x_i=x]$
<br/>
* Pick $\theta^\ast = \text{argmin}_\theta \, \mathbb{L}(\theta)$, and we are done! Right?
    * No...? &#128553;
    * $\mathbb{L}$ is non-convex

---
### Minimizing non-convex differentiable functions

* Iterative procedure:
    * Choose start $\theta_0$
    * In iteration $k$ update: $\theta_{k + 1} \leftarrow O(\theta_k, \mathbb{L})$
    * Stop if some criterion is met

* Gradient Descent:
    $$\theta_{k + 1} \leftarrow \theta_k - \eta  \cdot \nabla\mathbb{L}(\theta_k)$$

* Under mild conditions we get convergence to **local** minima

---
### How to Train a Neural Network

1) Choose the architecture
    1) Number of layers
    2) Number of neurons per layer
    3) Activation function
2) Choose loss function
3) Choose optimizer
4) Run the optimizer and hope for the best...

* Done &#127881; &#127881; &#127881; !!!

---

<!-- paginate: false -->
<!-- _class: lead -->

# Bonus (Gradient Descent)

---
###

---
<!-- _footer: Online repository: `github.com/timmens/neural-net` -->
### References
![bg width:300px right:30%](figures/qr_code_repo.png)

<style scoped>
ul li {
    font-size: 22px;
}
</style>

- Hastie, Tibshirani & Friedman (2009). The elements of statistical learning: data mining, inference, and prediction.

- Sanderson (2017). Neural Networks. (3Blue1Brown).

- Goodfellow, Bengio & Courville (2016). Deep Learning.

- Kidger & Lyons (2020). Universal Approximation with Deep Narrow Networks.

- Cybenko (1989).  Approximation by superpositions of a sigmoidal function.

- Hornik (1991). Approximation capabilities of multilayer feedforward networks.

- Pinkus (1999). Approximation theory of the MLP model in neural networks.
