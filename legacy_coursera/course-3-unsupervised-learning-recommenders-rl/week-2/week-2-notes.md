---
Topic: "Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning - Week 2"
Source: "Coursera / Andrew Ng"
---

# Core Notes

## Table of Contents
1.  [Introduction to Recommender Systems](#key-concept-1-introduction-to-recommender-systems)
2.  [Content-Based Filtering](#key-concept-2-content-based-filtering)
3.  [Collaborative Filtering](#key-concept-3-collaborative-filtering)
4.  [Matrix Factorization](#key-concept-4-matrix-factorization)

* **My Legend:**
    * 🔑 **Key Definition:** For critical, must-know vocabulary.
    * ❓ **Question:** For things you don't understand or want to explore later.
    * 🔗 **Connection:** For links to your existing knowledge.
    * 💡 **Insight:** For "aha!" moments or key takeaways.
    * ⚠️ **Warning:** For common mistakes, pitfalls, or important limitations.

---

## Key Concept 1: Introduction to Recommender Systems

-   🔑 **Key Definitions:**
    -   **Recommender System:** A system that seeks to predict the "rating" or "preference" a user would give to an item. They are widely used in applications like e-commerce, movie recommendations, and content personalization.
-   💡 **Insight:** Recommender systems are a key driver of user engagement and sales for many online businesses.

## Key Concept 2: Content-Based Filtering

-   🔑 **Key Definitions:**
    -   **Content-Based Filtering:** A type of recommender system that recommends items based on their attributes. It matches users with items that are similar to what they have liked in the past.
-   💡 **Insight:** Content-based filtering does not require data about other users. It relies solely on the user's own past behavior and the features of the items.
-   ❓ **Questions:**
    -   What is the main limitation of content-based filtering? (It can lead to a lack of diversity in recommendations, as it only recommends items similar to what the user already knows).

## Key Concept 3: Collaborative Filtering

-   🔑 **Key Definitions:**
    -   **Collaborative Filtering:** A type of recommender system that makes predictions about the interests of a user by collecting preferences from many users. The underlying assumption is that if two users have similar preferences in the past, they will have similar preferences in the future.
-   💡 **Insight:** Collaborative filtering can recommend items that are very different from what a user has seen before, leading to more diverse and serendipitous recommendations.
-   ⚠️ **Warnings:**
    -   Collaborative filtering suffers from the "cold start" problem: it's difficult to make recommendations for new users or new items with no rating history.

## Key Concept 4: Matrix Factorization

-   🔑 **Key Definitions:**
    -   **Matrix Factorization:** A powerful collaborative filtering technique that decomposes the user-item interaction matrix into the product of two lower-dimensionality matrices: a user-feature matrix and an item-feature matrix.
-   💡 **Insight:** Matrix factorization can learn "latent features" that describe users and items. For example, for movies, it might learn features like "genre," "actor," or "director" on its own, without being explicitly told.
-   🔗 **Connection:** Matrix factorization is a form of dimensionality reduction, similar to PCA, but applied to the user-item matrix.

---

# Module Summary

*(After finishing the module, write a 3-5 sentence summary here from memory **before** reviewing your notes above. This is a critical step for retention.)*

This week shifted focus to recommender systems, a powerful application of machine learning. I learned about the two main approaches: content-based filtering, which uses item features to make recommendations, and collaborative filtering, which leverages the preferences of similar users. A key and powerful collaborative filtering technique, matrix factorization, was introduced. This method can learn latent features of users and items, leading to highly personalized and effective recommendations.
