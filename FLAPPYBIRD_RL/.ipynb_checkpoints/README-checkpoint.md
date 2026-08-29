# 🐦 Flappy Bird - Deep Reinforcement Learning

This project implements a **Deep Reinforcement Learning agent** that learns to play Flappy Bird using a **Deep Q-Network (DQN)**.

Instead of manually controlling the bird, the agent learns which action to take by interacting with the game environment and receiving rewards.

## 🎯 Objective

The objective is to train an agent that learns to:

- Keep the bird alive
- Avoid pipes
- Pass through pipe gaps
- Maximize the game score

## 🧠 Algorithm

This project uses:

**Deep Q-Network (DQN)**

DQN combines:

- Reinforcement Learning
- Q-Learning
- Neural Networks
- Experience Replay

## 🔄 DQN Workflow

```text
Game Environment
       ↓
     State
       ↓
 Neural Network
       ↓
   Q-Values
       ↓
     Action
       ↓
Game Environment
       ↓
   Reward + Next State
       ↓
Experience Replay
       ↓
    Training