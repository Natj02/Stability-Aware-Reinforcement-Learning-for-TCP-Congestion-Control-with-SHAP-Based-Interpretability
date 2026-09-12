# Stability-Aware Reinforcement Learning for TCP Congestion Control with SHAP-Based Interpretability

Reference implementation for the master's thesis of the same title (Mapúa University, 2026).
A PPO agent for TCP congestion control, trained and evaluated in ns-3 via ns3-gym,
with a variance-aware reward and DeepSHAP-based interpretability.

## Overview

The system spans a C++/Python boundary bridged by ns3-gym:

- **ns-3 side (C++):** the TCP simulation and single-sender dumbbell topology (`sim.cc`).
- **Agent side (Python):** the Gym environment, PPO training, evaluation, and analysis
  (`experiments/agent/`).

The training loop passes a six-feature state vector from ns-3 to the PPO agent and
returns a continuous congestion-window action; after training, DeepSHAP attributes the
policy's decisions to its input features and a perturbation test validates those
attributions.

## Repository structure
