"""Statistical comparison of variants (Sec 3.4.4).

Per-seed pairing: each seed contributes one observation per metric, averaged
across the 8 scenarios. Metric family: throughput, coefficient of variation
(the stability metric, per Sec 4.3.2), jitter, packet loss. For each
(comparison, metric): Shapiro-Wilk normality -> paired t-test if normal else
Wilcoxon; effect size; Holm-Bonferroni over the four-metric family per
comparison.
"""
import glob
import numpy as np
import pandas as pd
from paths import STATS, TABLES
from scipy import stats

# CV replaces raw variance as the stability metric (Sec 4.3.2).
METRICS = ["mean_throughput_mbps", "cv_throughput", "jitter_ms", "packet_loss_pct"]
HIGHER_BETTER = {"mean_throughput_mbps"}          # proposed wins if higher
LOWER_BETTER = {"cv_throughput", "jitter_ms", "packet_loss_pct"}  # wins if lower


def per_seed(variant: str) -> pd.DataFrame:
    """One row per seed: metrics averaged across the 8 scenarios.
    CV is derived per seed as sqrt(mean throughput variance)/mean throughput."""
    fs = glob.glob(str(TABLES / f"metrics_{variant}_seed*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = df.groupby("seed").agg(
        mean_throughput_mbps=("mean_throughput_mbps", "mean"),
        jitter_ms=("jitter_ms", "mean"),
        packet_loss_pct=("packet_loss_pct", "mean"),
        _mt=("mean_throughput_mbps", "mean"),
        _vt=("throughput_variance_mbps2", "mean"),
    )
    g["cv_throughput"] = np.sqrt(g["_vt"]) / g["_mt"]
    return g[METRICS].sort_index().reset_index()


def cohens_d_paired(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def rank_biserial(a, b):
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def favors_proposed(metric, mean_prop, mean_other):
    if metric in HIGHER_BETTER:
        return mean_prop > mean_other
    return mean_prop < mean_other       # lower is better


def compare(a_name, b_name):
    a, b = per_seed(a_name), per_seed(b_name)
    merged = a.merge(b, on="seed", suffixes=(f"_{a_name}", f"_{b_name}"))
    rows, raw_p = [], []
    for m in METRICS:
        xa = merged[f"{m}_{a_name}"].to_numpy()
        xb = merged[f"{m}_{b_name}"].to_numpy()
        diff = xa - xb
        if np.allclose(diff, 0):
            rows.append(dict(metric=m, test="none", stat=0.0, p_raw=1.0,
                             effect=0.0, mean_a=xa.mean(), mean_b=xb.mean(),
                             favors_proposed=favors_proposed(m, xa.mean(), xb.mean())))
            raw_p.append(1.0); continue
        sh = stats.shapiro(diff).pvalue if len(diff) >= 3 else 0.0
        if sh > 0.05:
            t, p = stats.ttest_rel(xa, xb); test, eff = "paired-t", cohens_d_paired(diff)
        else:
            t, p = stats.wilcoxon(xa, xb); test, eff = "wilcoxon", rank_biserial(xa, xb)
        rows.append(dict(metric=m, test=test, stat=float(t), p_raw=float(p),
                         effect=float(eff), mean_a=float(xa.mean()),
                         mean_b=float(xb.mean()), shapiro_p=float(sh),
                         favors_proposed=favors_proposed(m, xa.mean(), xb.mean())))
        raw_p.append(float(p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(raw_p)
    out["sig_holm"] = out["p_holm"] < 0.05
    out.insert(0, "comparison", f"{a_name}_vs_{b_name}")
    return out


def main():
    comps = [("proposed", "vanilla"), ("proposed", "cubic"), ("proposed", "bbr")]
    all_out = pd.concat([compare(a, b) for a, b in comps], ignore_index=True)
    all_out.to_csv(STATS / "significance.csv", index=False)
    pd.set_option("display.width", 220)
    for c in all_out.comparison.unique():
        sub = all_out[all_out.comparison == c]
        print(f"\n=== {c} ===")
        print(sub[["metric","test","mean_a","mean_b","effect",
                   "p_raw","p_holm","sig_holm","favors_proposed"]]
              .round(4).to_string(index=False))
    print(f"\n-> {STATS/'significance.csv'}")


if __name__ == "__main__":
    main()