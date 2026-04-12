---
Topic: "Course 1: Supervised Machine Learning: Regression and Classification - Week 1"
Source: "Coursera / Andrew Ng"
---

# Week 1: Introduction to Supervised Learning

## Table of Contents
1.  [Introduction to Machine Learning](#1-introduction-to-machine-learning)
2.  [Linear Regression with One Variable](#2-linear-regression-with-one-variable)
3.  [The Cost Function (Squared Error)](#3-the-cost-function-squared-error)
4.  [Gradient Descent](#4-gradient-descent)

## 1. Introduction to Machine Learning
Machine Learning (ML) is the science of getting computers to act without being explicitly programmed.

### Two Main Types
1.  **Supervised Learning:** The algorithm learns from *labeled* data (input $x$, output $y$).
    *   **Regression:** Predicts a continuous number (e.g., housing prices).
    *   **Classification:** Predicts a discrete category (e.g., spam vs. not spam).
2.  **Unsupervised Learning:** The algorithm learns from *unlabeled* data (only input $x$).
    *   **Clustering:** Grouping similar data points (e.g., customer segmentation).
    *   **Dimensionality Reduction:** Compressing data.

## 2. Linear Regression with One Variable
This is the simplest form of regression, fitting a straight line to data.

*   **Notation:**
    *   $m$: Number of training examples.
    *   $x$'s: Input variables / features.
    *   $y$'s: Output variable / target.
    *   $(x^{(i)}, y^{(i)})$: The $i$-th training example.

*   **Hypothesis Function:**
    $$h_\theta(x) = \theta_0 + \theta_1 x$$ or $$f_{w,b}(x^{(i)}) = wx^{(i)}+b$$
    *   $\theta_0$: y-intercept. This is also called the parameter "bias" ($b$) or "intercept".
    *   $\theta_1$: slope. This is also called the parameter "weight" ($w$) or "coefficient".

## 3. The Cost Function (Squared Error)
To find the best parameters ($\theta_0, \theta_1$), we minimize the difference between predicted values and actual values.

*   **Formula:**
    $$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
*   **Goal:** Minimize $J(\theta_0, \theta_1)$.
*   **Intuition:** The $\frac{1}{2}$ is a convenience for the derivative calculation later. We average the squared errors.

## 4. Gradient Descent
An iterative algorithm to minimize any cost function $J$.

*   **Algorithm:**
    Repeat until convergence:
    $$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$
    *   $\alpha$ (Alpha): **Learning Rate**. Controls how big a step we take.
    *   The derivative term determines the direction of the step.

*   **For Linear Regression (Specific Update Rules):**
    *   $\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$
    *   $\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$
    *   *Note: Updates must be simultaneous.*

---

# Self-Assessment Quiz
<details>
<summary><strong>Q1: What is the difference between Regression and Classification?</strong></summary>
<br>
**Answer:**
Regression predicts a continuous numerical output (e.g., price, temperature), while Classification predicts a discrete category or label (e.g., yes/no, cat/dog/bird).
</details>

<details>
<summary><strong>Q2: What happens if the Learning Rate ($\alpha$) is too large?</strong></summary>
<br>
**Answer:**
Gradient descent may fail to converge or even diverge (overshoot the minimum and get worse).
</details>

<details>
<summary><strong>Q3: What happens if the Learning Rate ($\alpha$) is too small?</strong></summary>
<br>
**Answer:**
Gradient descent will work but will be very slow, taking many iterations to reach the minimum.
</details>

<details>
<summary><strong>Q4: In the cost function $J(\theta)$, what does $m$ represent?</strong></summary>
<br>
**Answer:**
$m$ represents the number of training examples in the dataset.
</details>
