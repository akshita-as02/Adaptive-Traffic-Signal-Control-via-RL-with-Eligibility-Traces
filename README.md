# Adaptive Traffic Signal Control via RL with Eligibility Traces

**CS 5180 -- Reinforcement Learning and Sequential Decision Making, Spring 2026**
Akshita Singh · Pranav Rajesh Nair
Northeastern University, Khoury College of Computer Sciences

---

## Problem

Train RL agents to minimize vehicle wait times at a simulated 4-way intersection,
framed as a **continuing MDP** with Poisson arrivals and a mandatory 3-step yellow
delay on every phase switch.

The yellow delay is the core challenge: when an agent decides to switch phases,
the benefit (shorter queues in the new direction) does not appear until 3 steps later.
One-step TD methods cannot bridge this gap efficiently; eligibility traces can.

**MDP specification:**

| Component | Definition |
|---|---|
| State | (q_N, q_S, q_E, q_W, phase) -- binned queues {0,1,2,3} x phase {0,1} |
| State space | 4^4 x 2 = 512 states |
| Actions | KEEP (hold current phase) or SWITCH (triggers 3-step yellow delay) |
| Arrivals | Poisson(lambda_i) per lane per step |
| Service | 1 vehicle/step/active green lane |
| Reward | -sum of binned queue lengths, in [-12, 0] |
| Discount | gamma = 0.99 (continuing task) |

---

## Algorithms

| Method | Type | Role |
|---|---|---|
| Fixed timing | Non-learning (30-step cycle) | Baseline |
| Longest queue | Non-learning (greedy heuristic) | Baseline |
| Q-learning (lambda=0) | Off-policy TD, one-step | Ablation |
| SARSA(lambda=0.9) | On-policy TD + traces | Ablation |
| **Q(lambda=0.9)** | **Off-policy + replacing traces** | **Primary (Akshita)** |
| **Actor-Critic** | **Policy gradient + dual traces** | **Primary (Pranav)** |

---

## Key Results

Results averaged over 3 independent training seeds (42, 7, 123). Format is mean +/- std.

### Final performance -- mean vehicle wait per step (lower is better)

| Method | Sym. light (0.3) | Sym. heavy (0.7) | Asym. rush (NS=0.8) |
|---|---|---|---|
| Fixed timing | 6.48 | 11.92 | 8.17 |
| Longest queue | 11.99 | 12.00 | 11.99 |
| Q-learning | 8.07 +/- 0.14 | 11.51 +/- 0.31 | 7.58 +/- 1.09 |
| SARSA(0.9) | 6.51 +/- 0.13 | 7.63 +/- 0.03 | 6.39 +/- 0.00 |
| **Q(lambda=0.9)** | **6.17 +/- 0.32** | **7.63 +/- 0.03** | **6.39 +/- 0.00** |
| **Actor-Critic** | **5.87 +/- 0.52** | **7.63 +/- 0.03** | **6.39 +/- 0.00** |

Actor-Critic achieves a mean wait of **5.87** on symmetric light -- a 9% improvement over fixed timing.
Both primary methods achieve a **22% reduction** vs. fixed timing on asymmetric rush hour (6.39 vs. 8.17).

The most important finding is the **stability gap**: all three trace-based methods converge to
exactly 6.39 with std=0.00 on rush hour (guaranteed convergence across all seeds), while
Q-learning achieves 7.58 +/- 1.09 -- seed 7 failed to converge entirely (7.32 vs. 6.39).
Eligibility traces are not just faster -- they are more reliable.

### Convergence speed -- asymmetric rush hour

| Method | Steps to reach 6.40 | Speedup |
|---|---|---|
| Q-learning | 90,000 | baseline |
| SARSA(0.9) | 20,000 | 4.5x faster |
| Q(lambda=0.9) | 10,000 | **9x faster** |
| Actor-Critic | 10,000 | **9x faster** |

### Lambda sensitivity (Q(lambda), asymmetric rush hour)

| lambda | Mean wait | Notes |
|---|---|---|
| 0.0 | 9.29 | Worse than fixed timing |
| 0.3 | 9.40 | Minimal improvement |
| 0.5 | 10.31 | **Worse than lambda=0** (non-monotonic) |
| 0.7 | 6.39 | Optimal floor reached |
| 0.9 | 6.39 | Optimal floor reached |
| 0.95 | 6.39 | Optimal floor reached |

lambda >= 0.7 required: the trace must retain 0.7^3 = 0.34 of its value across the
3-step yellow delay to correctly attribute credit to the switching decision.

---

## Repository Structure

```
Adaptive-Traffic-Signal-Control-via-RL-with-Eligibility-Traces/
|
|-- src/
|   |-- traffic_env.py          # MDP simulator (shared by all algorithms)
|   `-- tiles3.py               # Sutton's tile coding
|
|-- notebooks/
|   |-- cs5180_final_v2.ipynb   # All 6 methods + hyperparameter sweep + multi-seed eval
|   `-- baselines_combined_v4_2.ipynb  # Baselines + Q(lambda) standalone
|
|-- results/
|   |-- value_based_curves.png         # Q-learning vs SARSA vs Q(lambda)
|   |-- all_methods_final_curves.png   # All 4 RL methods vs baselines
|   |-- sweep_heatmaps.png             # Hyperparameter sweep heatmaps
|   `-- q_lambda_sensitivity.png       # Lambda sensitivity bar chart
|
`-- report/
    `-- paper_v5.tex                   # AAAI 2026 format final paper
```

---

## Setup

```bash
pip install numpy matplotlib
```

Place `traffic_env.py` and `tiles3.py` in the same directory as your notebook,
or add `src/` to your Python path.

**Google Colab / Vast.ai:**

```python
import sys
sys.path.insert(0, '/path/to/src/')
from traffic_env import TrafficEnv, KEEP, SWITCH
from tiles3 import IHT, tiles
```

---

## Quick Start

```python
from traffic_env import TrafficEnv, run_baseline, KEEP, SWITCH

# Asymmetric rush hour
env = TrafficEnv(mode="full", arrival_rates=[0.8, 0.8, 0.2, 0.2], seed=42)

# Non-learning baseline
stats = run_baseline("fixed", env, n_steps=20_000)
print(f"Fixed timing mean wait: {stats['mean_wait']:.4f}")  # ~8.17

# Step manually
state = env.reset()
for _ in range(1000):
    action = KEEP  # or your agent's choice
    next_state, reward, info = env.step(action)
    # info["yellow"] = True during yellow delay (action is ignored)
    # info["raw_q"] = raw un-binned queue lengths
```

**Tile coding setup (required for all RL algorithms):**

```python
from tiles3 import IHT, tiles

NUM_TILINGS = 8
IHT_SIZE    = 8192
Q_SCALE     = 8.0

iht = IHT(IHT_SIZE)

def get_features(ts, action):
    """ts = env.state_for_tiling() -- raw queues + phase."""
    return tiles(iht, NUM_TILINGS,
                 [ts[i] / Q_SCALE for i in range(4)],
                 ints=[action, int(ts[4])])
```

---

## Simulator Notes

- `env.state_for_tiling()` returns raw (un-binned) queue lengths. Pass these to tile
  coding -- passing binned values collapses adjacent states to the same tile.
- During yellow delay, `env.step(action)` silently ignores the agent's action.
- `simple` mode (2 lanes) is useful for debugging before scaling to `full` (4 lanes).
- All paper results use `full` mode, training seed 42, evaluation seed 99.

---

## Hyperparameter Summary

| Parameter | Q-learning | SARSA | Q(lambda) | Actor-Critic |
|---|---|---|---|---|
| alpha | 0.2/8 | 0.05/8 | 0.2/8 | w: 0.05/8, theta: 0.01/8 |
| lambda | n/a | 0.9 | 0.9 | 0.9 |
| epsilon | 0.1 (constant) | 0.1 to 0.01 (decay) | 0.1 to 0.01 (decay) | n/a (softmax) |
| trace clip | n/a | [-5, 5] | [-5, 5] | n/a |
| training steps | 150,000 | 150,000 | 150,000 | 150,000 |

---

## Paper

Akshita Singh and Pranav Rajesh Nair. "Adaptive Traffic Signal Control via
Reinforcement Learning with Eligibility Traces." CS 5180 Final Project,
Northeastern University, Spring 2026. AAAI 2026 format.

---

## Authors

**Akshita Singh** (singh.akshita@northeastern.edu)
Primary algorithm: Q(lambda=0.9) with Watkins's replacing trace

**Pranav Rajesh Nair** (nair.p2@northeastern.edu)
Primary algorithm: Actor-Critic with dual eligibility traces

Shared: MDP simulator (`traffic_env.py`), tile coding setup, all baselines, multi-seed evaluation

---

## References

- Sutton, R.S. and Barto, A.G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
  - Q(lambda): Section 12.10
  - Actor-Critic with traces: Section 13.6
  - Tile coding: Section 9.5.4
- Wiering, M.A. (2000). Multi-agent reinforcement learning for traffic light control. *ICML-2000*.
- Abdulhai, B., Pringle, R., and Karakoulas, G.J. (2003). Reinforcement learning for true adaptive
  traffic signal control. *Journal of Transportation Engineering* 129(3).
