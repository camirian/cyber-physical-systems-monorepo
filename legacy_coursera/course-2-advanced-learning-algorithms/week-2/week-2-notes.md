---
Topic: "Course 2: Advanced Learning Algorithms - Week 2"
Source: "Coursera / Andrew Ng"
---

# Core Notes

## Table of Contents
1.  [Training Neural Networks & Backpropagation](#key-concept-1-training-neural-networks--backpropagation)
2.  [Gradient Checking](#key-concept-2-gradient-checking)
3.  [Hyperparameter Tuning](#key-concept-3-hyperparameter-tuning)
4.  [Optimization Algorithms](#key-concept-4-optimization-algorithms)

* **My Legend:**
    * 🔑 **Key Definition:** For critical, must-know vocabulary.
    * ❓ **Question:** For things you don't understand or want to explore later.
    * 🔗 **Connection:** For links to your existing knowledge.
    * 💡 **Insight:** For "aha!" moments or key takeaways.
    * ⚠️ **Warning:** For common mistakes, pitfalls, or important limitations.

---

## Key Concept 1: Training Neural Networks & Backpropagation

-   🔑 **Key Definitions:**
    -   **Backpropagation:** The algorithm used to train neural networks. It calculates the gradient of the cost function with respect to the weights of the network, allowing us to update the weights using gradient descent.
    -   **Chain Rule:** The mathematical foundation of backpropagation. It's used to compute the derivatives of the cost function with respect to each weight in the network, layer by layer.
-   💡 **Insight:** Backpropagation is essentially "forward propagation" in reverse. After making a prediction (forward pass), you calculate the error and propagate it backward through the network to update the weights.

## Key Concept 2: Gradient Checking

-   🔑 **Key Definitions:**
    -   **Gradient Checking:** A method for verifying that your implementation of backpropagation is correct. It involves numerically approximating the gradients and comparing them to the gradients computed by backpropagation.
-   💡 **Insight:** Gradient checking is a crucial debugging tool for complex neural network implementations. It gives you confidence that your gradients are being calculated correctly.
-   ⚠️ **Warnings:**
    -   Gradient checking is computationally very expensive. It should only be used for debugging and should be turned off during training.

## Key Concept 3: Hyperparameter Tuning

-   🔑 **Key Definitions:**
    -   **Hyperparameters:** Parameters that are not learned by the model but are set before the training process begins. Examples include the learning rate, the number of hidden layers, the number of neurons per layer, and the regularization parameter.
-   💡 **Insight:** Finding the right hyperparameters is critical for getting good performance from a neural network. This is often an iterative process of trial and error.
-   ❓ **Questions:**
    -   What are some common strategies for hyperparameter tuning? (Grid search, random search, Bayesian optimization).

## Key Concept 4: Optimization Algorithms

-   🔑 **Key Definitions:**
    -   **Adam (Adaptive Moment Estimation):** A popular and effective optimization algorithm that adapts the learning rate for each parameter. It combines the ideas of Momentum and RMSprop.
    -   **Momentum:** An optimization algorithm that helps accelerate gradient descent in the relevant direction and dampens oscillations.
-   🔗 **Connection:** Adam is often the default choice for an optimizer because it works well on a wide range of problems and is less sensitive to the choice of learning rate than standard gradient descent.

---

# Module Summary

*(After finishing the module, write a 3-5 sentence summary here from memory **before** reviewing your notes above. This is a critical step for retention.)*

This week was all about how to train neural networks effectively. The core algorithm, backpropagation, was introduced as the method for calculating the gradients needed to update the network's weights. I learned how to verify my implementation of backpropagation using gradient checking. The important topic of hyperparameter tuning was also covered, along with more advanced optimization algorithms like Adam, which can significantly speed up the training process.
