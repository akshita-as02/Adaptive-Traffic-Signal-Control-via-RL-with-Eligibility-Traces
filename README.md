# Adaptive Traffic Signal Control via RL with Eligibility Traces

**CS 5180 - Reinforcement Learning, Spring 2026**  
Akshita Singh · Pranav Rajesh Nair  
Northeastern University, Khoury College of Computer Sciences

## Problem
Train RL agents to minimize vehicle wait times at a simulated 4-way 
intersection, framed as a continuing MDP with Poisson arrivals and 
a mandatory 3-step yellow delay on every phase switch.

## Algorithms
| Method | Type |
|---|---|
| Fixed timing | Non-learning baseline |
| Longest queue | Non-learning baseline |
| Q-learning (λ=0) | Off-policy TD, one-step |
| SARSA(λ=0.9) | On-policy TD + traces |
| Q(λ=0.9) | Off-policy + traces (primary)|
| Actor-Critic | Policy gradient + traces |

## Key Results (Asymmetric Rush Hour)
Fixed timing: 8.17 · Q-learning: 6.39 · Q(λ): **6.05** (symmetric light)

## Setup
```bash
pip install numpy matplotlib
# Place traffic_env.py and tiles3.py in the same directory as notebooks
```

## Files
- `src/traffic_env.py` — MDP simulator (shared by all algorithms)
- `src/tiles3.py` — Sutton's tile coding (Albus 1971)
- `notebooks/baselines_combined_v4_2.ipynb` — all baseline + Q(λ) experiments
