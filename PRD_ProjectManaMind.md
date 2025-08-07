# **Product Requirements Document: Project ManaMind**

**Author:** Gemini **Version:** 1.1 **Date:** August 7, 2025

### **1\. Overview & Vision**

**Project ManaMind** is an initiative to develop a state-of-the-art artificial intelligence agent capable of playing *Magic: The Gathering* (MTG) at a superhuman level. Inspired by the success of DeepMind's AlphaGo and AlphaZero, ManaMind will leverage deep reinforcement learning and self-play to master the immense strategic complexity of MTG.  
The ultimate vision is to create an agent that not only defeats the best human players but also discovers novel strategies, deck archetypes, and lines of play, thereby contributing to the collective understanding of the game and pushing the boundaries of AI research.

### **2\. The Problem**

*Magic: The Gathering* represents a "grand challenge" for artificial intelligence for several key reasons:

* **Immense State Space:** A single game involves a vast number of variables: cards in hand, on the battlefield, in graveyards, and in libraries; player life totals; mana availability; counters; and ongoing effects.  
* **Hidden Information:** Players have incomplete information due to the opponent's hand and the random order of cards in the library. The AI must make decisions under uncertainty.  
* **Dynamic & Evolving Environment:** The game is not static. New card sets are released every few months, constantly changing the metagame and requiring continuous adaptation.  
* **Complex Action Space:** At any point, a player can have hundreds of possible actions (playing lands, casting spells, activating abilities, attacking with various creatures, etc.), making the decision tree incredibly broad.

### **3\. Goals & Objectives**

Our goals are ambitious and phased, providing clear benchmarks for success.

* **P0 (Critical):** Develop an agent that can play valid, rule-abiding games of MTG within the **Forge game engine**.  
* **P1 (High):** Achieve a win rate greater than 80% against the built-in AI opponents in Forge.  
* **P1 (High):** Demonstrate successful integration with the *Magic: The Gathering Arena* (MTGA) client, capable of playing a full, valid game.  
* **P2 (Medium):** Reach Platinum rank on the MTGA competitive ladder.  
* **P2 (Medium):** Achieve Diamond and ultimately Mythic rank on the MTGA ladder.  
* **P3 (Low/Stretch):** Discover and popularize a novel, competitive deck archetype previously unknown to the human player base.  
* **P3 (Low/Stretch):** Defeat a current, high-profile MTG professional player in a publicized exhibition match.

### **4\. Target Audience**

1. **AI Research Community:** ManaMind will serve as a benchmark and case study for applying RL to games with imperfect information and dynamic environments.  
2. **The *Magic: The Gathering* Community:** Players, streamers, and content creators who are interested in high-level strategy, AI-driven analysis, and the future of gaming.  
3. **The Broader Gaming & Tech Industries:** A successful project will be a significant public demonstration of AI capabilities.

### **5\. Features & Requirements**

#### **5.1. Core AI Agent**

* **Game State Encoder:** A neural network module responsible for converting the complex game state into a fixed-size numerical vector (tensor).  
* **Reinforcement Learning Model:** An AlphaZero-style model architecture, consisting of:  
  * **Policy Network:** Outputs a probability distribution over all legal actions.  
  * **Value Network:** Outputs a scalar value predicting the probability of winning from the current state.  
* **Monte Carlo Tree Search (MCTS):** The agent will use MCTS, guided by the policy and value networks, to explore potential future game states and select the optimal move.  
* **Self-Play Loop:** The agent will learn primarily by playing millions of games against itself within the Forge engine.

#### **5.2. Game Engine Integration (Primary Training Environment)**

* **Forge Interface:** A robust software layer that allows the Python-based AI agent to communicate with the Java-based Forge game engine. This interface will handle starting games, sending actions, and receiving game state updates.  
* **Headless Operation:** The system must be able to run thousands of Forge instances in parallel without a graphical user interface (GUI) to maximize training speed.

#### **5.3. MTGA Integration (Deployment Environment)**

* **Game Client Interface:** A software layer that allows the trained AI to "see" the game state and "input" actions into the MTGA client, likely via screen-reading and input simulation. This will be used for evaluation and live play, not for initial training.

#### **5.4. Technical Foundation & Open-Source Strategy**

* **Game Engine:** We will build upon the **Forge** ([github.com/Card-Forge/forge](https://github.com/Card-Forge/forge)) open-source project. Forge provides a comprehensive, scriptable MTG rules engine, saving years of development effort in re-implementing game logic.  
* **Card Database:** We will use **MTGJSON** ([mtgjson.com](https://mtgjson.com/)) as the canonical source for all card data. This structured data is essential for the Game State Encoder to understand card properties and text.

#### **5.5. Training & Evaluation**

* **Distributed Training Infrastructure:** A scalable, cloud-based system to run thousands of self-play games in parallel using Dockerized Forge instances.  
* **Analytics Dashboard:** A web-based dashboard to visualize key metrics: training progress, win rate over time, and performance against specific decks or opponents.

### **6\. Roadmap & Milestones**

* **Phase 1: Foundation & Forge Integration (Target: 3-6 months)**  
  * **Focus:** Integrate the AI agent with the Forge engine and prove the learning loop.  
  * **Key Results:**  
    * Functional interface between the Python RL agent and the Java Forge engine.  
    * Agent can play a full, valid game within Forge.  
    * Initial self-play loop is operational and shows learning progress.  
    * **Goal:** Achieve \>80% win rate against built-in Forge AI.  
* **Phase 2: Mastery & MTGA Adaptation (Target: 6-12 months)**  
  * **Focus:** Achieve expert-level performance in Forge and adapt the agent to play on MTGA.  
  * **Key Results:**  
    * Massive scaling of the self-play infrastructure.  
    * Agent demonstrates expert-level play within the Forge environment.  
    * Develop the screen-reading and input simulation interface for MTGA.  
    * **Goal:** Agent can successfully play on the MTGA ladder and achieve Platinum rank.  
* **Phase 3: Superhuman Performance (Target: 12-24 months)**  
  * **Focus:** Push for top-tier performance on the MTGA ladder and explore novel strategies.  
  * **Key Results:**  
    * Agent consistently performs at a Mythic level on MTGA.  
    * Begin experiments with novel deck generation.  
    * Organize exhibition matches against human experts.  
    * **Goal:** Achieve a top 100 Mythic ranking.

### **7\. Risks & Open Questions**

* **Forge Integration Complexity:** The primary technical risk is building a stable, high-performance interface between the Python RL code and the Java-based Forge engine. **Mitigation:** Start with a simple proof-of-concept; dedicate focused engineering effort.  
* **MTGA Client Instability:** The lack of an official API means the MTGA interface will always be brittle and could break with any client update. **Mitigation:** De-risk by focusing all initial training on Forge. Treat the MTGA interface as a deployment target that requires ongoing maintenance.  
* **Computational Cost:** Training a model of this complexity will require significant computational resources. **Mitigation:** Secure budget and optimize the Forge-based training pipeline for efficiency.