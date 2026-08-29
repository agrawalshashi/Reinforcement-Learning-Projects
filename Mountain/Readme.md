# 🚗 MountainCar — Q-Learning

A Reinforcement Learning project implementing **Q-Learning** on the classic `MountainCar-v0` environment using **Gymnasium**.

---

## 🎯 Objective

The goal of the agent is to drive the car up the mountain and reach the flag at the target position.

The agent must learn:

- When to accelerate left
- When to accelerate right
- When to maintain its movement
- How to build enough momentum to reach the goal

---

## 🌍 Environment

The project uses:

```text
MountainCar-v0
```

from **Gymnasium**.

### State Space

The environment provides two continuous state variables:

```text
State = [Position, Velocity]
```

Approximate ranges:

```text
Position → [-1.2, 0.6]
Velocity → [-0.07, 0.07]
```

### Action Space

There are 3 possible actions:

| Action | Description |
|-------:|-------------|
| 0 | Accelerate Left |
| 1 | No Acceleration |
| 2 | Accelerate Right |

---

## 🧠 Why State Discretization?

Q-Learning traditionally uses a Q-table:

```text
Q(state, action)
```

However, MountainCar has continuous values for position and velocity.

For example:

```text
Position = -0.52731
Velocity = 0.00382
```

There can be a very large number of possible states.

Therefore, the continuous state space is divided into a fixed number of discrete bins.

For example:

```text
Position → 20 bins
Velocity → 20 bins

20 × 20 = 400 discrete states
```

The Q-table then becomes:

```text
Q[Position][Velocity][Action]
```

---

## 📐 Q-Learning

The agent updates its Q-values using the Q-Learning equation:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```

Where:

- `α` = Learning Rate
- `γ` = Discount Factor
- `r` = Reward
- `s` = Current State
- `a` = Current Action
- `s'` = Next State
- `a'` = Next Action

Q-Learning is an **off-policy reinforcement learning algorithm** because the agent learns using the best possible next action.

---

## 🎲 Exploration vs Exploitation

An **ε-greedy policy** is used for action selection.

Initially, the agent explores the environment more frequently.

As training progresses, epsilon is gradually reduced so that the agent relies more on the learned Q-values.

```text
High ε
  ↓
More Exploration
  ↓
Learning
  ↓
Lower ε
  ↓
More Exploitation
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|------:|
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.99 |
| Initial Epsilon | 1.0 |
| Minimum Epsilon | 0.01 |
| Epsilon Decay | 0.995 |
| Position Bins | 20 |
| Velocity Bins | 20 |
| Training Episodes | 10,000 |

These parameters can be adjusted to study their effect on learning.

---

## 📊 Training

During training, the agent:

1. Resets the environment.
2. Converts the continuous state into a discrete state.
3. Selects an action using the ε-greedy policy.
4. Performs the action.
5. Receives a reward.
6. Observes the next state.
7. Updates the Q-table.
8. Reduces epsilon.
9. Repeats until the episode ends.

---

## 📈 Results

Training performance can be analyzed using:

- Episode reward
- Moving average reward
- Epsilon decay
- Evaluation performance

A moving average of episode rewards is used to make the learning trend easier to observe.

### Example

```text
Episode Reward
      │
      │              ╭──────
      │          ╭───╯
      │      ╭───╯
      │  ╭───╯
      │──╯
      └──────────────────────
             Episodes
```

---

## 🧪 Evaluation

After training, the agent is evaluated with exploration disabled.

Instead of randomly selecting actions, the agent chooses:

```text
action = argmax(Q(state, action))
```

This allows the learned policy to be tested without unnecessary exploration.

---

## 💾 Saving the Q-Table

The trained Q-table can be saved using Python's `pickle` module.

Example:

```python
with open("mountaincar_q_table.pkl", "wb") as file:
    pickle.dump(q_table, file)
```

It can later be loaded using:

```python
with open("mountaincar_q_table.pkl", "rb") as file:
    q_table = pickle.load(file)
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Gymnasium
- Matplotlib
- Jupyter Notebook
- Q-Learning
- ε-Greedy Policy
- State Discretization

---

## 📂 Project Structure

```text
MountainCar/
│
├── MountainCar_Q_Learning.ipynb
├── mountaincar_q_table.pkl
└── README.md
```

---

## 🚀 Future Improvements

- Experiment with different numbers of state-discretization bins.
- Tune learning rate and discount factor.
- Compare Q-Learning with SARSA.
- Visualize the learned policy.
- Implement Deep Q-Learning (DQN).
- Compare tabular Q-Learning with DQN.

---

## 👨‍💻 Author

**Shashi Agrawal**

This project is part of my journey of learning and implementing **Reinforcement Learning** concepts through practical projects.