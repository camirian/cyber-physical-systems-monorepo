---
Topic: "Course 2: Advanced Learning Algorithms - Week 3"
Source: "Coursera / Andrew Ng"
---

# Core Notes

## Table of Contents
1.  [Decision Trees](#key-concept-1-decision-trees)
2.  [Building Decision Trees](#key-concept-2-building-decision-trees)
3.  [Tree Ensembles](#key-concept-3-tree-ensembles)
4.  [Advice for Applying Machine Learning](#key-concept-4-advice-for-applying-machine-learning)

* **My Legend:**
    * 🔑 **Key Definition:** For critical, must-know vocabulary.
    * ❓ **Question:** For things you don't understand or want to explore later.
    * 🔗 **Connection:** For links to your existing knowledge.
    * 💡 **Insight:** For "aha!" moments or key takeaways.
    * ⚠️ **Warning:** For common mistakes, pitfalls, or important limitations.

---

## Key Concept 1: Decision Trees

-   🔑 **Key Definitions:**
    -   **Decision Tree:** A tree-like model of decisions and their possible consequences. It's a type of supervised learning algorithm that is used for both classification and regression.
    -   **Root Node:** The topmost node in a decision tree.
    -   **Leaf Node:** A terminal node in a decision tree that represents a class label or a continuous value.
    -   **Splitting:** The process of dividing a node into two or more sub-nodes.
-   💡 **Insight:** Decision trees are easy to understand and interpret. The learned model can be visualized as a flowchart.

### Visualizing a Decision Tree Split
```mermaid
graph TD
    A[Is Ear Shape Pointy?]
    A -->|Yes| B[Cat]
    A -->|No| C[Is Face Round?]
    C -->|Yes| D[Dog]
    C -->|No| E[Other]
```

## Key Concept 2: Building Decision Trees

-   🔑 **Key Definitions:**
    -   **Information Gain:** A metric used to decide which feature to split on at each step in building the tree. It measures how much "information" a feature gives us about the class.
    -   **Gini Impurity:** Another metric used to measure the quality of a split in a decision tree.
-   ❓ **Questions:**
    -   How do you prevent a decision tree from overfitting? (By pruning the tree or setting a maximum depth).

## Key Concept 3: Tree Ensembles

-   🔑 **Key Definitions:**
    -   **Ensemble Learning:** A machine learning technique where multiple models are trained to solve the same problem and combined to get better results.
    -   **Random Forest:** An ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction. It uses bagging to create different training data for each tree.
    -   **Gradient Boosted Trees (XGBoost, LightGBM):** An ensemble method where new models are added to correct the errors made by existing models. The models are added sequentially.
-   💡 **Insight:** Ensemble methods are some of the most powerful techniques in machine learning and are often used to win machine learning competitions.

## Key Concept 4: Advice for Applying Machine Learning

-   🔑 **Key Definitions:**
    -   **Train/Dev/Test Sets:** Splitting your data into three sets: a training set to train the model, a development (or validation) set to tune hyperparameters, and a test set to evaluate the final performance of the model.
-   💡 **Insight:** Having separate dev and test sets is crucial to avoid "overfitting" to the test set and getting an overly optimistic estimate of your model's performance.
-   ⚠️ **Warnings:**
    -   Make sure your dev and test sets come from the same distribution.

---

# Module Summary

*(After finishing the module, write a 3-5 sentence summary here from memory **before** reviewing your notes above. This is a critical step for retention.)*

This week explored powerful non-linear models, moving beyond neural networks. I learned about decision trees, an intuitive and interpretable model, and how they are built using concepts like information gain. The main focus was on tree ensembles, which combine multiple decision trees to achieve high performance. I was introduced to Random Forests and Gradient Boosted Trees. Finally, the week covered practical advice for applying machine learning, such as the importance of splitting data into training, development, and test sets.
