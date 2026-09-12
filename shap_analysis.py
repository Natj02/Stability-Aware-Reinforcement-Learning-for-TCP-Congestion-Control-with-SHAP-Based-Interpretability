"""SHAP interpretability analysis for the proposed agent (Sec 3.4.4-3.4.6).

DeepExplainer on the actor's mean-action head. Global importance aggregated
across the 10 proposed seeds; local attributions, summary plot, and the
perturbation faithfulness test on a representative seed. Latency gradient is
zero-inflated by construction (obs[3] = 0 on single-sample intervals), so
global importance is reported both pooled and conditioned on non-zero gradient
(Sec 3.4.5).
"""
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from cc_env import FEATURES, RewardCfg, TcpCcEnv, eval_scenarios
from paths import DATA_EVAL, MODELS, SHAP_OUT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

FEAT_COLS = [f"f_{n}" for n in FEATURES]
GRAD_IDX = FEATURES.index("latency_gradient")   # the sparse feature
N_BACKGROUND = 100
N_EXPLAIN = 4000            # per seed; total foreground = 10 * this, subsampled
SEEDS = list(range(10))
REP_SEED = 0               # representative seed for local/summary/faithfulness


class ActorMeanNet(nn.Module):
    """Standalone features -> mean action, extracted from the SB3 policy.

    DeepExplainer needs a plain tensor->tensor module; the SB3 policy returns a
    distribution. This wraps the deterministic mean-action path only.
    """
    def __init__(self, policy):
        super().__init__()
        self.extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, x):
        latent_pi, _ = self.extractor(x)
        return self.action_net(latent_pi)      # mean action (pre-squash)


def load_actor(seed: int):
    model_dir = MODELS / f"delta2.5_N20_seed{seed}"
    model = PPO.load(str(model_dir / "final"), device="cpu")
    net = ActorMeanNet(model.policy).eval()
    # VecNormalize stats to transform raw features into the space the net sees
    dummy = DummyVecEnv([lambda: TcpCcEnv(RewardCfg(delta=2.5, n_var=20),
                                          fixed_link=eval_scenarios()["bw_low"])])
    vn = VecNormalize.load(str(model_dir / "vecnorm.pkl"), dummy)
    dummy.close()
    return net, vn


def raw_features(seed: int) -> np.ndarray:
    """The 6 raw (pre-normalization) features from this seed's eval telemetry."""
    path = DATA_EVAL / f"proposed_seed{seed}_telemetry.csv"
    df = pd.read_csv(path)
    df = df[df["watchdog"] == 0]                 # drop dead frames
    return df[FEAT_COLS].to_numpy(np.float32)


def normalize(vn: VecNormalize, x: np.ndarray) -> np.ndarray:
    """Apply VecNormalize obs transform (the space the actor computes on)."""
    return np.clip((x - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def shap_for_seed(seed: int, n_explain: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (shap_values, explained_inputs_normalized) for one seed."""
    net, vn = load_actor(seed)
    raw = raw_features(seed)
    norm = normalize(vn, raw)

    bg_idx = rng.choice(len(norm), size=min(N_BACKGROUND, len(norm)), replace=False)
    fg_idx = rng.choice(len(norm), size=min(n_explain, len(norm)), replace=False)
    background = torch.tensor(norm[bg_idx])
    foreground = torch.tensor(norm[fg_idx])

    explainer = shap.DeepExplainer(net, background)
    sv = explainer.shap_values(foreground, check_additivity=False)
    sv = np.asarray(sv).reshape(len(fg_idx), len(FEATURES))   # (n, 6)
    return sv, raw[fg_idx]                                     # raw for sparsity mask


def global_importance():
    """Mean |SHAP| per feature, pooled and conditional, aggregated over seeds."""
    rng = np.random.default_rng(20260804)
    per_seed_pooled, per_seed_cond = [], []
    rep_sv = rep_raw = None

    for s in SEEDS:
        sv, raw_fg = shap_for_seed(s, N_EXPLAIN, rng)
        per_seed_pooled.append(np.abs(sv).mean(axis=0))                  # (6,)
        nz = raw_fg[:, GRAD_IDX] != 0.0                                  # non-zero grad
        cond = np.abs(sv[nz]).mean(axis=0) if nz.sum() else np.full(6, np.nan)
        per_seed_cond.append(cond)
        if s == REP_SEED:
            rep_sv, rep_raw = sv, raw_fg
        print(f"  seed {s}: {nz.sum()}/{len(sv)} non-zero-gradient intervals "
              f"({100*nz.mean():.0f}%)")

    pooled = np.vstack(per_seed_pooled)      # (10, 6)
    cond = np.vstack(per_seed_cond)          # (10, 6)

    tbl = pd.DataFrame({
        "feature": FEATURES,
        "pooled_mean": pooled.mean(0), "pooled_sd": pooled.std(0, ddof=1),
        "conditional_mean": np.nanmean(cond, 0),
        "conditional_sd": np.nanstd(cond, axis=0, ddof=1),
    })
    tbl["pooled_rank"] = tbl["pooled_mean"].rank(ascending=False).astype(int)
    tbl["conditional_rank"] = tbl["conditional_mean"].rank(ascending=False).astype(int)
    tbl = tbl.sort_values("pooled_mean", ascending=False).reset_index(drop=True)
    return tbl, rep_sv, rep_raw


def plot_global(tbl):
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(tbl))
    ax.barh(y - 0.2, tbl["pooled_mean"], 0.4, xerr=tbl["pooled_sd"],
            label="pooled", capsize=3)
    ax.barh(y + 0.2, tbl["conditional_mean"], 0.4, xerr=tbl["conditional_sd"],
            label="non-zero gradient only", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(tbl["feature"]); ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|  (aggregated over 10 seeds)")
    ax.set_title("Global feature importance — proposed agent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SHAP_OUT / "global_importance.png", dpi=150)
    plt.close(fig)


def plot_summary(rep_sv, rep_raw):
    """Beeswarm on the representative seed (raw feature values on colour)."""
    shap.summary_plot(rep_sv, rep_raw, feature_names=list(FEATURES), show=False)
    plt.tight_layout()
    plt.savefig(SHAP_OUT / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_local(rep_sv, rep_raw, n=3):
    """Waterfall for a few representative decisions."""
    base = 0.0
    idxs = np.linspace(0, len(rep_sv) - 1, n).astype(int)
    for k, i in enumerate(idxs):
        expl = shap.Explanation(values=rep_sv[i], base_values=base,
                                data=rep_raw[i], feature_names=list(FEATURES))
        shap.plots.waterfall(expl, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_OUT / f"local_decision_{k}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()


def faithfulness(tbl, seed=REP_SEED, n=2000):
    """Top-k vs bottom-k perturbation test (Sec 3.4.6).

    Perturb the k most / least SHAP-important features (per the pooled ranking),
    measure mean absolute change in the actor's action. Faithful iff perturbing
    important features moves the action more than perturbing unimportant ones.
    """
    net, vn = load_actor(seed)
    raw = raw_features(seed)
    norm = normalize(vn, raw)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(norm), size=min(n, len(norm)), replace=False)
    X = torch.tensor(norm[idx])
    with torch.no_grad():
        base_act = net(X).numpy().ravel()

    order = tbl.sort_values("pooled_mean", ascending=False)["feature"].tolist()
    fi = [FEATURES.index(f) for f in order]      # feature indices, most->least

    def perturb_and_measure(cols):
        Xp = X.clone().numpy()
        # replace chosen columns with a random permutation (break their signal)
        for c in cols:
            Xp[:, c] = np.random.permutation(Xp[:, c])
        with torch.no_grad():
            act = net(torch.tensor(Xp)).numpy().ravel()
        return float(np.mean(np.abs(act - base_act)))

    res = {
        "top1": perturb_and_measure(fi[:1]),
        "top2": perturb_and_measure(fi[:2]),
        "bottom1": perturb_and_measure(fi[-1:]),
        "bottom2": perturb_and_measure(fi[-2:]),
    }
    res["faithful_top1_vs_bottom1"] = res["top1"] > res["bottom1"]
    res["faithful_top2_vs_bottom2"] = res["top2"] > res["bottom2"]
    return res


def main():
    global SEEDS, N_EXPLAIN
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10,
                   help="how many proposed seeds to aggregate (default 10)")
    p.add_argument("--n-explain", type=int, default=N_EXPLAIN)
    a = p.parse_args()

    SEEDS = list(range(a.seeds))
    N_EXPLAIN = a.n_explain
    SHAP_OUT.mkdir(parents=True, exist_ok=True)

    print("Computing SHAP across seeds...")
    tbl, rep_sv, rep_raw = global_importance()
    tbl.to_csv(SHAP_OUT / "global_importance.csv", index=False)
    print("\n=== GLOBAL FEATURE IMPORTANCE ===")
    print(tbl.round(4).to_string(index=False))

    plot_global(tbl)
    plot_summary(rep_sv, rep_raw)
    plot_local(rep_sv, rep_raw)

    print("\n=== FAITHFULNESS (perturbation test) ===")
    faith = faithfulness(tbl)
    for k, v in faith.items():
        print(f"  {k}: {v}")
    pd.DataFrame([faith]).to_csv(SHAP_OUT / "faithfulness.csv", index=False)

    print(f"\n-> outputs in {SHAP_OUT}")


if __name__ == "__main__":
    main()
