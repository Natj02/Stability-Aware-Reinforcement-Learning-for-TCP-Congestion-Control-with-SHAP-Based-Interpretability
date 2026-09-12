"""Sensitivity analysis for delta and N (Sec 3.2.2).

Trains short jobs across a delta x N grid, evaluates each on validation links
sampled from the training distribution (disjoint from training draws and from
the 8 held-out test scenarios), and applies the rejection rule: reject any
delta whose variance reduction drops mean throughput below the delta=0 ablation
by more than a tolerance. Among accepted configs, pick the one with the lowest
throughput variance.

Selection runs are short (400k steps): we are ranking delta, not measuring
final performance. The chosen (delta, N) is then trained to 4M in production.
"""
import argparse
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from cc_env import EP_STEPS, Link, RewardCfg, TcpCcEnv, sample_link
from evaluate import episode_metrics, run_episode
from paths import DATA_SWEEP, MODELS, STATS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---- sweep grid -----------------------------------------------------------
DELTAS = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0]
NS = [10, 20]
SWEEP_SEEDS = [0, 1, 2]
SWEEP_STEPS = 400_000
VAL_LINKS_PER = 6          # validation links per (delta, N, seed)
VAL_EPISODES = 2          # episodes per validation link
TAU = 0.05                # max tolerated throughput loss vs delta=0

# Validation links: fixed stream, disjoint from training (different RNG seed)
# and from eval_scenarios (which are hand-set corners, not random draws).
_val_rng = np.random.default_rng(999_983)
VAL_LINKS = [sample_link(_val_rng) for _ in range(VAL_LINKS_PER)]


def train_one(delta: float, n_var: int, seed: int, port_base: int) -> Path:
    tag = f"sweep_delta{delta:g}_N{n_var}_seed{seed}"
    out = MODELS / tag
    if (out / "final.zip").exists():
        return out                                   # resume: skip done runs
    subprocess.run(
        ["python", "train.py",
         "--seed", str(seed), "--delta", str(delta), "--n-var", str(n_var),
         "--timesteps", str(SWEEP_STEPS), "--n-envs", "4",
         "--out", str(out)],
        cwd=str(Path(__file__).parent), check=True)
    return out


def eval_on_val(model_dir: Path, delta: float, n_var: int,
                seed: int, port: int) -> list[dict]:
    rw = RewardCfg(delta=delta, n_var=n_var)
    model = PPO.load(str(model_dir / "final"), device="cpu")
    vecnorm = model_dir / "vecnorm.pkl"
    recs = []
    for li, link in enumerate(VAL_LINKS):
        venv = DummyVecEnv([lambda l=link: TcpCcEnv(reward=rw, seed=seed,
                                                    port=port, fixed_link=l)])
        if vecnorm.exists():
            venv = VecNormalize.load(str(vecnorm), venv)
            venv.training = False
            venv.norm_reward = False
        for ep in range(VAL_EPISODES):
            meta = dict(variant="sweep", seed=seed,
                        scenario=f"val{li}", episode=ep)
            m = run_episode(model, venv, None, meta)     # no telemetry writer
            recs.append({"delta": delta, "n_var": n_var, "seed": seed,
                         "val_link": li, "episode": ep, **m})
        venv.close()
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port-base", type=int, default=5700)
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing sweep models, only re-evaluate")
    a = p.parse_args()

    recs = []
    for di, (delta, n_var) in enumerate(itertools.product(DELTAS, NS)):
        if delta == 0.0 and n_var != NS[0]:
            continue                                  # N is inert at delta=0
        for seed in SWEEP_SEEDS:
            model_dir = MODELS / f"sweep_delta{delta:g}_N{n_var}_seed{seed}"
            if not a.skip_train:
                model_dir = train_one(delta, n_var, seed, a.port_base)
            port = a.port_base + di * 10 + seed
            recs.extend(eval_on_val(model_dir, delta, n_var, seed, port))

    df = pd.DataFrame(recs)
    df.to_csv(DATA_SWEEP / "sweep_raw.csv", index=False)

    # aggregate: mean throughput and variance per (delta, N) across seeds/links
    agg = (df.groupby(["delta", "n_var"])
             .agg(mean_tput=("mean_throughput_mbps", "mean"),
                  mean_var=("throughput_variance_mbps2", "mean"),
                  tput_sd=("mean_throughput_mbps", "std"))
             .reset_index())

    # rejection rule: throughput must stay within TAU of the delta=0 ablation
    base_tput = agg.loc[agg.delta == 0.0, "mean_tput"].iloc[0]
    agg["tput_floor"] = (1.0 - TAU) * base_tput
    agg["accepted"] = agg["mean_tput"] >= agg["tput_floor"]

    agg = agg.sort_values(["accepted", "mean_var"],
                          ascending=[False, True]).reset_index(drop=True)
    agg.to_csv(STATS / "sensitivity.csv", index=False)

    print(agg.round(4).to_string(index=False))
    print(f"\ndelta=0 ablation throughput: {base_tput:.3f} Mbps")
    print(f"acceptance floor (>= {(1-TAU)*100:.0f}%): {agg['tput_floor'].iloc[0]:.3f} Mbps")

    accepted = agg[agg.accepted]
    if len(accepted) and not (accepted.delta == 0.0).all():
        # best = lowest variance among accepted, non-trivial (delta>0) configs
        nontrivial = accepted[accepted.delta > 0.0]
        best = (nontrivial.iloc[0] if len(nontrivial) else accepted.iloc[0])
        print(f"\nSELECTED: delta={best.delta:g}, N={int(best.n_var)}")
        print(f"  variance {best.mean_var:.4g} Mbps^2 "
              f"(vs {agg.loc[agg.delta==0,'mean_var'].iloc[0]:.4g} at delta=0)")
        print(f"  throughput {best.mean_tput:.3f} Mbps "
              f"({100*best.mean_tput/base_tput:.1f}% of ablation)")
    else:
        print("\nNo delta>0 config passed the throughput floor. "
              "Consider lowering delta range or raising TAU.")


if __name__ == "__main__":
    main()
