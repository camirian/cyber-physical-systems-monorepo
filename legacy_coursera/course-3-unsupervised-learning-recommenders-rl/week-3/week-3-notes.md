---
Topic: "Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning - Week 3"
Source: "Coursera / Andrew Ng"
---

# Core Notes

## Table of Contents
1.  [Introduction to Reinforcement Learning](#key-concept-1-introduction-to-reinforcement-learning)
2.  [Markov Decision Processes (MDPs)](#key-concept-2-markov-decision-processes-mdps)
3.  [Value Functions and Policies](#key-concept-3-value-functions-and-policies)
4.  [Q-Learning](#key-concept-4-q-learning)

* **My Legend:**
    * 🔑 **Key Definition:** For critical, must-know vocabulary.
    * ❓ **Question:** For things you don't understand or want to explore later.
    * 🔗 **Connection:** For links to your existing knowledge.
    * 💡 **Insight:** For "aha!" moments or key takeaways.
    * ⚠️ **Warning:** For common mistakes, pitfalls, or important limitations.

---

## Key Concept 1: Introduction to Reinforcement Learning

-   🔑 **Key Definitions:**
    -   **Reinforcement Learning (RL):** A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a cumulative reward.
    -   **Agent:** The learner or decision-maker.
    -   **Environment:** The world in which the agent exists and interacts.
    -   **State:** A snapshot of the environment at a particular time.
    -   **Action:** A move the agent can make in the environment.
    -   **Reward:** The feedback the agent receives from the environment after taking an action in a particular state.
-   💡 **Insight:** RL is about learning through trial and error, much like how humans and animals learn. The agent is not told what to do, but instead must discover which actions yield the most reward.

## Key Concept 2: Markov Decision Processes (MDPs)

-   🔑 **Key Definitions:**
    -   **Markov Decision Process (MDP):** A mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. It's the formal way to describe the RL environment.
    -   **Markov Property:** The assumption that the future is independent of the past, given the present. In other words, the current state contains all the necessary information to make a decision.
-   🔗 **Connection:** MDPs provide the theoretical foundation for reinforcement learning.

## Key Concept 3: Value Functions and Policies

-   🔑 **Key Definitions:**
    -   **Policy (π):** The agent's strategy or behavior. It's a mapping from states to actions.
    -   **Value Function (V(s)):** A function that estimates the expected cumulative reward an agent will receive starting from a particular state 's' and following a policy π.
    -   **Q-Value Function (Q(s, a)):** A function that estimates the expected cumulative reward an agent will receive starting from a state 's', taking an action 'a', and then following a policy π.
-   💡 **Insight:** The Q-value function is often more useful than the value function because it tells us the value of taking a specific action in a specific state, which makes it easier to choose the best action.

## Key Concept 4: Q-Learning

-   🔑 **Key Definitions:**
    -   **Q-Learning:** A model-free reinforcement learning algorithm that learns a policy by learning the Q-values for each state-action pair. It does not require a model of the environment (i.e., it doesn't need to know the transition probabilities between states).
-   💡 **Insight:** Q-learning is a powerful and widely used RL algorithm. It can learn the optimal policy even if the agent starts with a random policy.
-   ⚠️ **Warnings:**
    -   Q-learning can be slow to converge, especially in large state-action spaces.

---

# Module Summary

*(After finishing the module, write a 3-5 sentence summary here from memory **before** reviewing your notes above. This is a critical step for retention.)*

This week introduced the exciting field of reinforcement learning (RL), where an agent learns to make optimal decisions through trial and error. I learned the core components of an RL problem: the agent, environment, state, action, and reward. The mathematical framework for RL, the Markov Decision Process (MDP), was introduced. A key focus was on value functions and policies, which represent the agent's knowledge and behavior. Finally, I was introduced to Q-learning, a powerful algorithm for learning an optimal policy without needing a model of the environment.
