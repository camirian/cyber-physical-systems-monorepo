---
Topic: "Course 1: Supervised Machine Learning: Regression and Classification - Week 2"
Source: "Coursera / Andrew Ng"
---

# Week 2: Regression with Multiple Input Variables

## Table of Contents
1.  [Multiple Linear Regression](#1-multiple-linear-regression)
2.  [Vectorization](#2-vectorization)
3.  [Gradient Descent for Multiple Regression](#3-gradient-descent-for-multiple-regression)
4.  [Practical Tips for Gradient Descent](#4-practical-tips-for-gradient-descent)
5.  [Polynomial Regression](#5-polynomial-regression)

## 1. Multiple Linear Regression
Extending linear regression to handle multiple features (variables).

*   **Notation:**
    *   $n$: Number of features.
    *   $x_j$: The $j$-th feature.
    *   $\vec{x}$: Feature vector (including $x_0=1$).
    *   $\vec{w}$: Parameter vector (weights).
    *   $b$: Bias term (formerly $\theta_0$).

*   **Hypothesis Function (Vectorized):**
    $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

## 2. Vectorization
Using linear algebra libraries (like NumPy) to compute predictions for all examples simultaneously, rather than using `for` loops. This is significantly faster.

*   **Python Example:**
    ```python
    f = np.dot(w, x) + b  # Much faster than a loop
    ```

## 3. Gradient Descent for Multiple Regression
The update rule is the same, but applied to each parameter $w_j$.

$$w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) x_j^{(i)}$$
$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})$$

## 4. Practical Tips for Gradient Descent

### Feature Scaling
If features have very different ranges (e.g., House Size: 0-2000, Bedrooms: 1-5), gradient descent will oscillate and be slow.
*   **Goal:** Scale features to roughly $-1 \le x_j \le 1$.
*   **Mean Normalization:** $x_j := \frac{x_j - \mu_j}{max - min}$
*   **Z-score Normalization:** $x_j := \frac{x_j - \mu_j}{\sigma_j}$

### Learning Rate ($\alpha$)
*   **Debugging:** Plot $J(\vec{w},b)$ vs. Number of Iterations. It should decrease every iteration.
*   **Choice:**
    *   If $J$ increases: $\alpha$ is too big.
    *   If $J$ decreases very slowly: $\alpha$ is too small.
    *   Try values like: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1...

## 5. Polynomial Regression
We can fit non-linear curves by creating new features from existing ones.
*   Example: If we have feature $x$ (size), we can create $x^2$ (size squared) or $x^3$ to fit a curve.
*   $$f(x) = w_1 x + w_2 x^2 + w_3 x^3 + b$$
*   *Important:* Feature scaling is critical here because $x^2$ and $x^3$ will have huge ranges.

---

# Self-Assessment Quiz
<details>
<summary><strong>Q1: Why is Vectorization preferred over `for` loops?</strong></summary>
<br>
**Answer:**
Vectorization allows the computer to use parallel hardware (SIMD) to perform calculations much faster than sequential loops. It also leads to shorter, cleaner code.
</details>

<details>
<summary><strong>Q2: Why do we perform Feature Scaling?</strong></summary>
<br>
**Answer:**
To ensure all features have a similar range (e.g., -1 to 1). This makes the cost function "rounder" (contours look like circles rather than elongated ellipses), allowing Gradient Descent to converge much faster.
</details>

<details>
<summary><strong>Q3: If the Cost Function $J$ is increasing over iterations, what should you do?</strong></summary>
<br>
**Answer:**
The learning rate $\alpha$ is likely too large. You should decrease it.
</details>

<details>
<summary><strong>Q4: True or False: In Polynomial Regression, you are changing the model from Linear Regression to something else.</strong></summary>
<br>
**Answer:**
**False.** It is still Linear Regression, but the *features* are non-linear (e.g., $x^2$). The model is still linear with respect to the parameters $w$.
</details>
