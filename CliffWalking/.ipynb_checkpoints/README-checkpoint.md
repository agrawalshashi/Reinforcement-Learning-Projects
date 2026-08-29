# 🧗 Cliff Walking - Reinforcement Learning

This project implements Reinforcement Learning algorithms on the **Cliff Walking** environment.

The main goal is to train an agent to find the optimal path from the starting position to the goal while avoiding the cliff.

## 🚀 Algorithms Implemented

- Q-Learning
- SARSA

## 🎯 Objective

The agent starts from a fixed position and must reach the destination while avoiding the cliff.

The environment provides a negative reward for every step and a large negative reward for falling into the cliff.

The agent learns an optimal policy through repeated interaction with the environment.

## 🧠 Reinforcement Learning Concepts

This project demonstrates:

- Q-Table
- Exploration vs Exploitation
- Epsilon-Greedy Policy
- Learning Rate
- Discount Factor
- Temporal Difference Learning
- Q-Learning
- SARSA
- Reward-based learning

## 🔵 Q-Learning

Q-Learning is an **off-policy Temporal Difference learning algorithm**.

The Q-value is updated using:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]

Where:

- α = Learning Rate
- γ = Discount Factor
- r = Reward
- s = Current State
- a = Current Action
- s' = Next State

## 🟠 SARSA

SARSA is an **on-policy Temporal Difference learning algorithm**.

The update rule is:

Q(s,a) ← Q(s,a) + α [r + γQ(s',a') - Q(s,a)]

SARSA considers the action that the agent actually chooses in the next state.

## ⚔️ Q-Learning vs SARSA

| Feature | Q-Learning | SARSA |
|---|---|---|
| Type | Off-policy | On-policy |
| Next Action | Best possible action | Actual selected action |
| Policy | Learns optimal policy | Learns policy being followed |
| Exploration | ε-Greedy | ε-Greedy |

## 📊 Results

The agent learns a policy that allows it to reach the goal while avoiding the cliff.

Training performance can be analyzed using:

- Episode rewards
- Number of steps
- Learned Q-values
- Final policy

## 🛠️ Technologies

- Python
- NumPy
- Gymnasium
- Matplotlib
- Jupyter Notebook

## 📂 Project Structure

```text
CliffWalking/
│
├── Q_Learning.ipynb
├── SARSA.ipynb
└── README.md