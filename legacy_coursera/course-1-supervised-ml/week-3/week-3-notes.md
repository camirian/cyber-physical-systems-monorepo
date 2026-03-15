---
Topic: "Course 1: Supervised Machine Learning: Regression and Classification - Week 3"
Source: "Coursera / Andrew Ng"
---

# Week 3: Classification & Overfitting

## Table of Contents
1.  [Logistic Regression](#1-logistic-regression)
2.  [Cost Function for Logistic Regression](#2-cost-function-for-logistic-regression)
3.  [The Problem of Overfitting](#3-the-problem-of-overfitting)
4.  [Regularization](#4-regularization)

## 1. Logistic Regression
Used for **Classification** problems where $y \in \{0, 1\}$.
*   **Sigmoid Function:**
    $$g(z) = \frac{1}{1 + e^{-z}}$$
    *   Outputs values between 0 and 1.
    *   Interpreted as probability: $P(y=1 | x; \vec{w}, b)$.

*   **Hypothesis:**
    $$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b)$$

*   **Decision Boundary:**
    *   Predict $y=1$ if $f(x) \ge 0.5$ (which means $z \ge 0$).
    *   Predict $y=0$ if $f(x) < 0.5$ (which means $z < 0$).

## 2. Cost Function for Logistic Regression
We cannot use Squared Error because it results in a non-convex function (many local minima).
*   **Loss Function (Binary Cross-Entropy):**
    $$L(f_{\vec{w},b}(\vec{x}), y) = -y \log(f_{\vec{w},b}(\vec{x})) - (1-y) \log(1 - f_{\vec{w},b}(\vec{x}))$$
*   **Cost Function:**
    $$J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})$$

## 3. The Problem of Overfitting
*   **Underfitting (High Bias):** Model is too simple to capture the trend. High training error, high test error.
*   **Overfitting (High Variance):** Model is too complex, memorizing noise. Low training error, high test error.
*   **Generalization:** The ability of the model to perform well on new, unseen data.

### Addressing Overfitting
1.  **Collect more data:** Helps the model learn the true underlying pattern.
2.  **Select features:** Remove irrelevant features (Feature Selection).
3.  **Regularization:** Reduce the magnitude of parameters $w_j$.

## 4. Regularization
Add a penalty term to the cost function to discourage large weights.

*   **Regularized Cost Function:**
    $$J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} L(...) + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$
    *   $\lambda$ (Lambda): **Regularization Parameter**.
    *   If $\lambda$ is very large $\rightarrow$ Underfitting (weights $\approx$ 0).
    *   If $\lambda$ is 0 $\rightarrow$ Overfitting (standard logistic regression).

*   **Gradient Descent with Regularization:**
    $$w_j := w_j - \alpha [\frac{1}{m} \sum ... + \frac{\lambda}{m} w_j]$$
    *   The term $\frac{\lambda}{m} w_j$ shrinks the weight on every step.

---

# Self-Assessment Quiz
<details>
<summary><strong>Q1: Why can't we use the Squared Error cost function for Logistic Regression?</strong></summary>
<br>
**Answer:**
Using Squared Error with the Sigmoid function results in a **non-convex** cost function with many local minima, making it difficult for Gradient Descent to find the global minimum.
</details>

<details>
<summary><strong>Q2: If a model has very low training error but very high test error, what is it suffering from?</strong></summary>
<br>
**Answer:**
**Overfitting (High Variance).** The model has memorized the training data but fails to generalize to new data.
</details>

<details>
<summary><strong>Q3: How does increasing the regularization parameter $\lambda$ affect the model?</strong></summary>
<br>
**Answer:**
Increasing $\lambda$ increases the penalty on large weights, forcing the model to be simpler. This reduces variance (overfitting) but increases bias (underfitting).
</details>

<details>
<summary><strong>Q4: True or False: We usually regularize the bias term $b$.</strong></summary>
<br>
**Answer:**
**False.** By convention, we usually only regularize the weights $w_1, ..., w_n$ and leave the bias term $b$ unregularized, although regularizing $b$ makes little difference in practice.
</details>
