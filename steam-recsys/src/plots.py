"""Figures for the report. Colours: fixed categorical order (blue, orange, aqua, ...),
one-hue sequential blue for the heatmap; text stays in neutral ink."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.axisbelow": True,
})


def k_selection(rows, best_k, path):
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ks, [r["sse"] for r in rows], color=SERIES[0], lw=2, marker="o", ms=6)
    axes[0].set(title="Elbow: within-cluster SSE", xlabel="k", ylabel="SSE")
    axes[1].plot(ks, [r["silhouette"] for r in rows], color=SERIES[0], lw=2, marker="o", ms=6)
    axes[1].axvline(best_k, color=INK2, lw=1, ls="--")
    axes[1].annotate(f"chosen k = {best_k}", (best_k, np.nanmax([r["silhouette"] for r in rows])),
                     xytext=(6, -12), textcoords="offset points", color=INK2)
    axes[1].set(title="Silhouette (sampled users)", xlabel="k", ylabel="silhouette")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def cluster_heatmap(centers, vocab, sizes, path, n_show=15, diverging=False):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    if diverging:
        # z-scored centroids: most extreme tags across clusters, blue = below avg, orange = above
        order = np.argsort(-np.abs(centers).max(0))[:n_show]
        M = centers[:, order]
        lim = max(np.abs(M).max(), 1e-6)
        cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f4a27f", "#b8461b"])
        norm, label, title = TwoSlopeNorm(0, -lim, lim), "z-score vs. average user", \
            "Cluster centroids: tags played more (orange) / less (blue) than average"
    else:
        order = np.argsort(-centers.mean(0))[:n_show]
        M = centers[:, order]
        cmap, norm = LinearSegmentedColormap.from_list("seq", SEQ), None
        label, title = "tag share", "Cluster centroids: share of playtime per tag"
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * n_show, 1.2 + 0.45 * len(centers)))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(order)), [vocab[j] for j in order], rotation=45, ha="right")
    ax.set_yticks(range(len(centers)), [f"C{c} (n={sizes[c]})" for c in range(len(centers))])
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=label)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def tradeoff(grid, acc_key, tail_key, marks, path):
    """Every weight combination on validation: accuracy vs. long-tail recall."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.scatter(grid[acc_key], grid[tail_key], s=10, color="#b5b3ad", label="weight combinations", zorder=2)
    for c, (name, row) in enumerate(marks.items()):
        # decreasing size so identical picks stay visible as concentric rings
        ax.scatter([row[acc_key]], [row[tail_key]], s=[260, 150, 50][c % 3], color=SERIES[c], edgecolor=SURFACE,
                   linewidth=1, label=name, zorder=3 + c)
    ax.set(title="Hybrid weights on validation: accuracy vs. long-tail recall",
           xlabel=acc_key, ylabel=tail_key.replace("_group2", " (tail games)"))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def metric_bars(res, metric, path, title):
    names = list(res.keys())
    vals = [res[n][metric] for n in names]
    errs = [res[n].get(metric + "_ci95", 0) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.42 * len(names)))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, xerr=errs, color=SERIES[0], height=0.62,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:.4f}", va="center", ha="left", color=INK2, fontsize=9)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=f"{metric} (mean ± 95% CI over test users)")
    ax.set_xlim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def metric_dots(res, metric, path, title):
    """Dot + 95% CI per model on a zoomed axis (small differences, e.g. ablations)."""
    names = list(res.keys())
    vals = np.array([res[n][metric] for n in names])
    errs = np.array([res[n].get(metric + "_ci95", 0) for n in names])
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.45 * len(names)))
    y = np.arange(len(names))[::-1]
    ax.errorbar(vals, y, xerr=errs, fmt="o", color=SERIES[0], ecolor=INK2, elinewidth=1,
                capsize=3, ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for yi, v, e in zip(y, vals, errs):
        ax.annotate(f"{v:.4f}", (v + e, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.grid(axis="y", visible=False)
    pad = (vals + errs).max() - (vals - errs).min()
    ax.set_xlim((vals - errs).min() - 0.15 * pad, (vals + errs).max() + 0.35 * pad)
    ax.set(title=title, xlabel=f"{metric} (mean ± 95% CI; axis zoomed, not from zero)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def group_recall(res, k, group_names, path):
    names = list(res.keys())
    G = len(group_names)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / G
    x = np.arange(len(names))
    for g in range(G):
        vals = [res[n].get(f"recall@{k}_group{g}", np.nan) for n in names]
        ax.bar(x + (g - (G - 1) / 2) * w, vals, width=w * 0.92, color=SERIES[g], label=group_names[g])
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set(title=f"Recall@{k} by game popularity group", ylabel=f"recall@{k}")
    ax.legend(frameon=False, ncol=G, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def weight_sensitivity(rows, metric, path):
    """NDCG on validation as each weight varies (others at their tuned value)."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for c, (key, label) in enumerate([("w_km", "K-means"), ("w_ar", "Apriori"), ("w_cb", "Content")]):
        pts = rows.get(key, [])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[c], lw=2, marker="o", ms=6, label=label)
    ax.set(title="Re-rank weight sensitivity (validation, others at tuned value)",
           xlabel="component weight", ylabel=metric)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
