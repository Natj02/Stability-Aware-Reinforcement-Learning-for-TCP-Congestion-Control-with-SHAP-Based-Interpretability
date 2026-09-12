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

/ns3/
sim_rl-cc-train.cc Main simulation. Driven by cc_env.py for all training and
evaluation. Includes the CUBIC/BBR baseline telemetry path
(--telemetryOut). This is the file that reproduces Chapter 4.
sim_rl-cc.cc Frozen pilot config. Used only to generate the Table 3.3.2
pilot telemetry sample; not used for the main results.

/experiments/agent/
cc_env.py ns3-gym environment: 6-feature layer + stability-aware reward.
Spawns one ns-3 process per episode against sim_rl-cc-train.
train.py PPO training (Stable-Baselines3), domain-randomized links.
evaluate.py Deterministic evaluation over the 8 held-out scenarios;
writes per-interval telemetry + per-episode metrics.
baselines.py CUBIC / BBR runners (ns-3 built-in implementations).
shap_analysis.py DeepSHAP global/local attributions + faithfulness test.
stats.py Paired testing (Shapiro-Wilk gate, t / Wilcoxon, Holm).
cv_check.py Coefficient-of-variation stability computation.
sweep_delta.py Reward sensitivity grid over (δ, N_v).
paths.py Path configuration for data/model/result directories.

/configs/ Hyperparameters and scenario definitions.
env-lock.txt Exact package versions used for the reported results.



## Requirements

- ns-3 **3.40** with the ns3-gym (opengym) contrib module
- Python 3.x (see `env-lock.txt` for the exact version and all package pins)
- Key Python packages: Stable-Baselines3, PyTorch (CUDA build), ns3gym, shap,
  pandas, NumPy, SciPy
- CUDA-capable GPU for training (evaluation runs on CPU)

Install the Python environment from the lockfile:

```bash
pip install -r env-lock.txt      # adjust if env-lock.txt is a pip freeze / conda export
```

## Building the simulation

Place the simulation in the ns3-gym examples tree and build:

```bash
# copy the main sim into an ns-3 example folder
cp ns3/sim_rl-cc-train.cc \
   <ns-3.40>/contrib/opengym/examples/rl-cc-train/sim.cc

cd <ns-3.40>
./ns3 build
```

The Python environment (`cc_env.py`) spawns the compiled binary directly and ns3gym
attaches to the listening port.

## Reproducing the results

Run from `experiments/agent/`. Training is repeated across 10 seeds; the commands below
show one seed as a template.

**1. Train the proposed agent (δ = 2.5) and the vanilla ablation (δ = 0):**

```bash
python train.py --variant proposed --seed 0 --delta 2.5 --n-var 20
python train.py --variant vanilla  --seed 0 --delta 0.0
```

**2. Evaluate all four variants over the 8 held-out scenarios:**

```bash
python evaluate.py --variant proposed --seed 0 --model-dir models/<tag>
python evaluate.py --variant vanilla  --seed 0 --model-dir models/<tag>
python evaluate.py --variant cubic     --seed 0        # baseline, no model
python evaluate.py --variant bbr       --seed 0        # baseline, no model
```

**3. Statistical comparison and stability metrics:**

```bash
python stats.py
python cv_check.py
```

**4. SHAP interpretability and faithfulness test:**

```bash
python shap_analysis.py
```

**5. Reward sensitivity grid (optional; reproduces Table 4.2.1):**

```bash
python sweep_delta.py
```

## Notes on measurement

- Throughput is time-weighted per interval and clipped to bottleneck capacity before
  aggregation. Short monitoring intervals closed by a batch of acknowledgments can make
  bytes/interval momentarily exceed the physical link rate; clipping removes this
  finite-interval artifact so the variance metric reflects genuine instability.
- Baseline (CUBIC/BBR) telemetry is logged by the C++ tracer (`--telemetryOut`) sampled
  at per-RTT cadence, matching the interval granularity the RL agent observes, so all
  four variants are compared over the same measurement basis.


## Citation

If you use this code, please cite:

[]
