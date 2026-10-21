"""Regenerates the report figures from the saved logs (no model training).
Usage: python make_figures.py        (reads logs/*.csv, writes figures/*.pdf)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figures", exist_ok=True)

def load(name, keep_na=True):
    p = os.path.join("logs", name)
    if not os.path.exists(p):
        print("SKIP (missing):", p); return None
    # the depth label "None" must stay a string, not become NaN
    return pd.read_csv(p) if keep_na else pd.read_csv(p, keep_default_na=False)

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join("figures", name)); plt.close(fig); print("wrote figures/" + name)

# ---------------- Q1 ----------------
g = load("q1a_grid.csv", keep_na=False)
if g is not None:
    g["max_depth"] = g["max_depth"].astype(str)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for m in sorted(g.min_samples_leaf.unique()):
        s = g[g.min_samples_leaf == m]
        ax[0].plot(s.max_depth, s.train_acc, "o--", label=f"train msl={m}"); ax[0].plot(s.max_depth, s.val_acc, "s-", label=f"val msl={m}")
        ax[1].plot(s.max_depth, s.n_leaves, "o-", label=f"msl={m}"); ax[2].plot(s.max_depth, s.frac_singleton_leaves, "o-", label=f"msl={m}")
    for a, t in zip(ax, ["accuracy", "#leaves", "fraction of single-example leaves"]):
        a.set_ylabel(t); a.set_xlabel("max_depth"); a.grid(alpha=.3)
    ax[0].legend(fontsize=6); ax[1].legend(fontsize=7); save(fig, "q1a_depth.pdf")

b = load("q1bc_bagging.csv")
if b is not None:
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    ax[0].plot(b.B, b.val_acc, "o-", label="val"); ax[0].plot(b.B, b.oob_acc, "s-", label="OOB"); ax[0].set_ylabel("accuracy"); ax[0].legend()
    ax[1].plot(b.B, b.oob_acc, "s-", color="C1"); ax[1].set_ylabel("OOB accuracy")
    ax[2].plot(b.B, b.disagreement_val, "o-", color="C2"); ax[2].set_ylabel("avg pairwise disagreement (val)")
    ax[3].plot(b.B, b.margin_val, "o-", color="C3"); ax[3].set_ylabel("mean vote margin (val)")
    for a in ax: a.set_xscale("log"); a.set_xlabel("B"); a.grid(alpha=.3)
    save(fig, "q1c_oob_disagree_margin.pdf")

# ---------------- Q2 ----------------
d = load("q2b_selected_diagnostics.csv")
if d is not None:
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
    for a, (c, t) in zip(ax, [("weighted_train_error", "weighted training error"), ("alpha", "learner weight alpha_t"),
                              ("max_sample_weight", "max sample weight"), ("top10_weight", "weight of 10 heaviest images")]):
        a.plot(d["round"], d[c]); a.set_title(t, fontsize=9); a.set_xlabel("round"); a.grid(alpha=.3)
    ax[2].axhline(1 / 240, ls=":", c="k"); ax[3].axhline(10 / 240, ls=":", c="k"); save(fig, "q2b_diagnostics.pdf")

g2 = load("q2b_grid.csv")
if g2 is not None:
    fig, a = plt.subplots(figsize=(4.5, 3.2))
    for dd in sorted(g2.depth.unique()):
        s = g2[g2.depth == dd]; a.plot(s.rounds, s.val_acc, "o-", label=f"depth {dd}")
    a.set_xlabel("rounds"); a.set_ylabel("validation accuracy"); a.legend(); a.grid(alpha=.3); save(fig, "q2b_val_grid.pdf")

# ---------------- Q3 ----------------
a3 = load("q3a_selected_models.csv")
if a3 is not None:
    t3 = a3.iloc[:3]; names = ["tree", "bagging", "boosting"]; x = range(3)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar([i - .19 for i in x], t3.bias2, .38, label="bias^2"); ax[0].bar([i + .19 for i in x], t3.variance, .38, label="variance")
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(names); ax[0].legend(); ax[0].set_title("measured bias^2 and variance", fontsize=9)
    ax[1].bar(list(x), t3.acc_mean, yerr=t3.acc_std, capsize=4); ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names)
    ax[1].set_title("clean test accuracy, mean +- sd over refits", fontsize=9); save(fig, "q3a_bias_variance.pdf")

b3 = load("q3b_depth_sweep.csv", keep_na=False)
if b3 is not None:
    b3["depth"] = b3["depth"].astype(str)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for col, a, t in [("bias2", ax[0], "bias^2"), ("variance", ax[1], "variance"), ("acc_mean", ax[2], "clean test accuracy")]:
        for model, mk in [(m, "o-" if "single" in m else "s--") for m in b3.model.unique()]:
            s = b3[b3.model == model]; a.plot(s.depth, s[col], mk, label=model)
        a.set_title(t, fontsize=9); a.set_xlabel("max_depth"); a.grid(alpha=.3)
    ax[0].legend(fontsize=7); save(fig, "q3b_depth_sweep.pdf")

d3 = load("q3d_final_table.csv"); pr = load("q3_per_refit_accuracy.csv")
if d3 is not None and pr is not None:
    nm = ["tree", "bagging", "boosting"]; xx = range(3)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3))
    ax[0].boxplot([pr[f"{n}_clean_acc"] for n in nm] + [pr[f"{n}_masked_acc"] for n in nm])
    ax[0].set_xticklabels([f"{n}\nclean" for n in nm] + [f"{n}\nmasked" for n in nm]); ax[0].tick_params(axis="x", labelsize=6); ax[0].set_title("test accuracy over refits", fontsize=9)
    ax[1].bar([i - .2 for i in xx], d3.bias2, .4, label="bias^2"); ax[1].bar([i + .2 for i in xx], d3.variance, .4, label="variance")
    ax[1].set_xticks(list(xx)); ax[1].set_xticklabels(nm); ax[1].legend(fontsize=7)
    ax[2].bar([i - .2 for i in xx], d3.acc_drop, .4, label="accuracy drop"); ax[2].bar([i + .2 for i in xx], d3.prediction_flip_rate, .4, label="prediction flip rate")
    ax[2].set_xticks(list(xx)); ax[2].set_xticklabels(nm); ax[2].legend(fontsize=7); ax[2].set_title("sensitivity to the masked block", fontsize=9)
    save(fig, "q3d_stability_sensitivity.pdf")
