"""End-to-end pipeline (v2): data -> subset -> split -> tuning on validation ->
refit on train+val -> final test evaluation, analyses, tables and figures.

    python run_pipeline.py --data-dir data/raw --out-dir results_v2
"""
import argparse
import itertools
import json
import logging
import os
import platform
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src import data as D
from src import preprocess as P
from src import kmeans as km
from src import plots
from src.metrics import evaluate, RankAccumulator
from src.models import Popularity, ClusterPop, RuleRec, ItemCF, ContentBased, RerankHybrid

log = logging.getLogger("pipeline")
GROUP_NAMES = ["head (top 20%)", "mid", "tail (bottom 50%)"]
TAIL = 2


def popularity_groups(X):
    """0 = head (top 20% of games by positive count), 2 = tail (bottom 50%), 1 = mid."""
    cnt = np.asarray((X > 0).sum(axis=0)).ravel()
    order = np.argsort(-cnt, kind="stable")
    g = np.ones(len(cnt), int)
    g[order[: int(0.2 * len(cnt))]] = 0
    g[order[int(0.5 * len(cnt)):]] = 2
    return g, cnt


def users_with_targets(T):
    return np.flatnonzero(np.diff(T.indptr) > 0)


def timed(msg):
    class _T:
        def __enter__(self):
            self.t = time.time(); log.info("%s ...", msg); return self

        def __exit__(self, *a):
            self.sec = time.time() - self.t; log.info("%s done (%.1fs)", msg, self.sec)
    return _T()


def make_kmeans(cfg, k):
    return ClusterPop(k, cfg.n_tags, cfg.kmeans_n_init, cfg.kmeans_max_iter, cfg.seed,
                      cfg.generic_tag_frac, cfg.feature_scaling)


def make_rules(cfg, support):
    return RuleRec(support, cfg.min_confidence, cfg.min_lift, cfg.max_itemset_len,
                   cfg.max_candidates, log)


def fit_all(cfg, ds, X, k, support, neighbors):
    return {"pop": Popularity().fit(X),
            "km": make_kmeans(cfg, k).fit(X, ds.item_tags),
            "ar": make_rules(cfg, support).fit(X),
            "cf": ItemCF(neighbors).fit(X),
            "cb": ContentBased(cfg.use_description, cfg.desc_weight, cfg.max_desc_features)
                  .fit(X, ds.item_tags, ds.item_desc)}


def rule_coverage(ar, users, batch):
    fired = 0
    for b0 in range(0, len(users), batch):
        fired += int((ar.score(users[b0:b0 + batch], tie_break=False).max(1) > 0).sum())
    return fired / max(len(users), 1)


def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    fig_dir = os.path.join(cfg.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))
    rng = np.random.default_rng(cfg.seed)
    ev = dict(k_list=cfg.k_list, batch_size=cfg.batch_size)
    tk = f"ndcg@{cfg.tune_k}"
    tail_key = f"recall@{max(cfg.k_list)}_group{TAIL}"

    # ------------------------------------------------------------------ data
    with timed("preprocessing (src/preprocess.py)"):
        reviews, titles, meta, _ = P.run(cfg, cfg.out_dir)
        ds = D.build_dataset_from_reviews(reviews, titles, meta)
        del reviews
    data_stats = D.describe(ds)
    miss = D.missing_metadata_report(ds)
    miss.to_csv(os.path.join(cfg.out_dir, "missing_metadata.csv"), index=False)
    data_stats["games_without_metadata_row"] = int((~ds.item_has_meta).sum())
    data_stats["games_with_empty_tags_but_row"] = int((miss.metadata_row_exists & miss.no_tags).sum())
    json.dump(data_stats, open(os.path.join(cfg.out_dir, "data_stats.json"), "w"), indent=2)
    log.info("subset: %s", data_stats)

    X_tr = ds.confidence(["train"])
    seen_tr = ds.seen(["train"])
    T_val = ds.targets("val")
    val_users = users_with_targets(T_val)
    tune_users = np.sort(rng.choice(val_users, min(cfg.tune_users, len(val_users)), replace=False))
    groups_tr, cnt_tr = popularity_groups(X_tr)
    tuning = []

    # ------------------------------------------- tuning on validation (fit = train)
    # (1) K-means: choose k by silhouette on train-only user features
    with timed("K-means k selection"):
        vocab, generic = km.top_tags(ds.item_tags, cfg.n_tags, cfg.generic_tag_frac)
        F, _ = km.user_features(X_tr, km.item_tag_matrix(ds.item_tags, vocab), cfg.feature_scaling)
        k_rows, best_k = km.select_k(F, cfg.k_candidates, cfg.kmeans_n_init, cfg.kmeans_max_iter,
                                     cfg.silhouette_sample, cfg.seed)
    log.info("generic tags removed from K-means features: %s", generic)
    pd.DataFrame(k_rows).to_csv(os.path.join(cfg.out_dir, "kmeans_k_selection.csv"), index=False)
    plots.k_selection(k_rows, best_k, os.path.join(fig_dir, "kmeans_k_selection.png"))
    m = make_kmeans(cfg, best_k).fit(X_tr, ds.item_tags)
    s, _ = evaluate(m.score, tune_users, T_val, seen_tr, item_group=groups_tr, **ev)
    tuning.append({"component": "kmeans", "param": f"k={best_k}", **s})

    # (2) Apriori: min_support
    best_ar, best_sup = None, None
    for sup in cfg.support_grid:
        with timed(f"Apriori min_support={sup}") as t:
            m = make_rules(cfg, sup).fit(X_tr)
        s, _ = evaluate(m.score, tune_users, T_val, seen_tr, item_group=groups_tr, **ev)
        tuning.append({"component": "apriori", "param": f"min_support={sup}",
                       "n_frequent_itemsets": len(m.freq), "n_rules": len(m.rules),
                       "rule_user_coverage": rule_coverage(m, tune_users, cfg.batch_size),
                       "fit_sec": round(t.sec, 2), **s})
        if best_ar is None or s[tk] > best_ar:
            best_ar, best_sup = s[tk], sup

    # (3) Item-based CF: neighbourhood size
    best_cf, best_nb = None, None
    for nb in cfg.cf_neighbors_grid:
        with timed(f"ItemCF neighbors={nb}") as t:
            m = ItemCF(nb).fit(X_tr)
        s, _ = evaluate(m.score, tune_users, T_val, seen_tr, item_group=groups_tr, **ev)
        tuning.append({"component": "itemcf", "param": f"neighbors={nb}", "fit_sec": round(t.sec, 2), **s})
        if best_cf is None or s[tk] > best_cf:
            best_cf, best_nb = s[tk], nb

    # (4) Content-based and the popularity baseline (no hyper-parameter search)
    with timed("fit tuned components on train"):
        comps = fit_all(cfg, ds, X_tr, best_k, best_sup, best_nb)
    for key in ["cb", "pop"]:
        s, _ = evaluate(comps[key].score, tune_users, T_val, seen_tr, item_group=groups_tr, **ev)
        tuning.append({"component": key, "param": "-", **s})
    pd.DataFrame(tuning).to_csv(os.path.join(cfg.out_dir, "tuning_val.csv"), index=False)

    # Hybrid: CF top-N candidates re-ranked; grid over (w_km, w_ar, w_cb), CF weight = 1
    cold_tr = cnt_tr < np.quantile(cnt_tr, cfg.cold_quantile)
    hyb = RerankHybrid(comps["cf"], comps, seen_tr, cfg.n_candidates, cold_tr, cfg.cold_boost)
    with timed("hybrid weight grid (re-rank)"):
        preps = []
        cand_hits = cand_tot = 0
        for b0 in range(0, len(tune_users), cfg.batch_size):
            ub = tune_users[b0:b0 + cfg.batch_size]
            p = hyb.prepare(ub)
            Tb = T_val[ub]
            for row in range(len(ub)):
                t = Tb.indices[Tb.indptr[row]:Tb.indptr[row + 1]]
                cand_hits += np.isin(t, p["cand"][row][p["valid"][row]]).sum()
                cand_tot += len(t)
            preps.append((ub, p))
        candidate_recall = cand_hits / max(cand_tot, 1)
        grid_rows = []
        for w_km, w_ar, w_cb in itertools.product(cfg.weight_grid, repeat=3):
            w = {"km": w_km, "ar": w_ar, "cb": w_cb}
            acc = RankAccumulator(cfg.k_list, ds.n_items, groups_tr)
            for ub, p in preps:
                acc.add(RerankHybrid.ranked_items(p, w, max(cfg.k_list)), T_val[ub])
            s, _ = acc.result()
            grid_rows.append({"w_km": w_km, "w_ar": w_ar, "w_cb": w_cb, **s})
    log.info("CF top-%d candidates contain %.1f%% of validation targets (upper bound for re-ranking)",
             cfg.n_candidates, 100 * candidate_recall)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(cfg.out_dir, "hybrid_weight_grid_val.csv"), index=False)

    best = grid.sort_values([tk, tail_key], ascending=False).iloc[0]
    ok = grid[grid[tk] >= cfg.tail_tolerance * best[tk]]
    tail_best = ok.sort_values([tail_key, tk], ascending=False).iloc[0]
    w_acc = {k: float(best[f"w_{k}"]) for k in RerankHybrid.keys}
    w_tail = {k: float(tail_best[f"w_{k}"]) for k in RerankHybrid.keys}
    cf_row = grid[(grid.w_km == 0) & (grid.w_ar == 0) & (grid.w_cb == 0)].iloc[0]

    sens = {}
    for key in ["w_km", "w_ar", "w_cb"]:
        o1, o2 = [o for o in ["w_km", "w_ar", "w_cb"] if o != key]
        sel = grid[(grid[o1] == best[o1]) & (grid[o2] == best[o2])]
        sens[key] = sorted(zip(sel[key], sel[tk]))
    plots.weight_sensitivity(sens, tk, os.path.join(fig_dir, "hybrid_weight_sensitivity.png"))
    plots.tradeoff(grid, tk, tail_key,
                   {"CF only": cf_row, "Hybrid (accuracy)": best, "Hybrid (tail-aware)": tail_best},
                   os.path.join(fig_dir, "hybrid_tradeoff_val.png"))
    chosen = {"k": int(best_k), "min_support": best_sup, "cf_neighbors": int(best_nb),
              "n_candidates": cfg.n_candidates, "weights_accuracy": w_acc, "weights_tail_aware": w_tail,
              "val_cf_only": {tk: float(cf_row[tk]), tail_key: float(cf_row[tail_key])},
              "val_hybrid_accuracy": {tk: float(best[tk]), tail_key: float(best[tail_key])},
              "val_hybrid_tail_aware": {tk: float(tail_best[tk]), tail_key: float(tail_best[tail_key])},
              "val_candidate_recall": candidate_recall}
    json.dump(chosen, open(os.path.join(cfg.out_dir, "chosen_params.json"), "w"), indent=2)
    log.info("chosen on validation: %s", chosen)

    # ---------------------------------------- final: refit on train+val, test once
    X_fit = ds.confidence(["train", "val"])
    seen_fit = ds.seen(["train", "val"])
    T_test = ds.targets("test")
    test_users = users_with_targets(T_test)
    with timed("refit on train+val"):
        comps = fit_all(cfg, ds, X_fit, best_k, best_sup, best_nb)
    groups, cnt = popularity_groups(X_fit)
    cold = cnt < np.quantile(cnt, cfg.cold_quantile)

    def hybrid(w):
        return RerankHybrid(comps["cf"], comps, seen_fit, cfg.n_candidates, cold, cfg.cold_boost, w).score

    # ablation of the accuracy-tuned hybrid; if it kept no component, ablate the tail-aware one
    base_name, base_w = ("Hybrid (accuracy)", w_acc) if any(w_acc.values()) else ("Hybrid (tail-aware)", w_tail)
    models = {
        "Popularity (baseline)": comps["pop"].score,
        "K-means cluster-pop": comps["km"].score,
        "Apriori rules": comps["ar"].score,
        "Item-based CF": comps["cf"].score,
        "Content-based": comps["cb"].score,
        "Hybrid (accuracy)": hybrid(w_acc),
        "Hybrid (tail-aware)": hybrid(w_tail),
    }
    labels = {"km": "K-means", "ar": "Apriori", "cb": "Content"}
    for key in RerankHybrid.keys:
        if base_w[key] > 0:
            models[f"{base_name} − {labels[key]}"] = hybrid({**base_w, key: 0.0})
    results, per_user = {}, {}
    for name, fn in models.items():
        with timed(f"test: {name}") as t:
            s, per = evaluate(fn, test_users, T_test, seen_fit, item_group=groups, **ev)
        s["eval_sec"] = round(t.sec, 2)
        results[name], per_user[name] = s, per
    res_df = pd.DataFrame(results).T
    res_df.index.name = "model"
    res_df.to_csv(os.path.join(cfg.out_dir, "metrics_test.csv"))

    # paired significance on per-user NDCG (same users for every model)
    sig = []
    for h in ["Hybrid (accuracy)", "Hybrid (tail-aware)"]:
        a = np.asarray(per_user[h][tk])
        for name in models:
            if name == h:
                continue
            b = np.asarray(per_user[name][tk])
            w = stats.wilcoxon(a, b, zero_method="zsplit") if np.any(a != b) else None
            sig.append({"comparison": f"{h} vs {name}", "mean_diff": a.mean() - b.mean(),
                        "wilcoxon_p": w.pvalue if w else 1.0, "n_users": len(a)})
    pd.DataFrame(sig).to_csv(os.path.join(cfg.out_dir, "significance_test.csv"), index=False)

    # ------------------------------------------------------- component analyses
    kmf = comps["km"]
    sizes = np.bincount(kmf.labels, minlength=best_k)
    rows = []
    for c in range(best_k):
        over = np.argsort(-kmf.centers[c])[:5]
        under = np.argsort(kmf.centers[c])[:3]
        members = np.flatnonzero(kmf.labels == c)
        lift = kmf.P[c] / np.maximum(kmf.global_pop, 1e-9) * (kmf.P[c] >= 0.05)
        rows.append({"cluster": c, "users": int(sizes[c]),
                     "over_represented_tags": ", ".join(f"{kmf.vocab[j]} ({kmf.centers[c, j]:+.2f})" for j in over),
                     "under_represented_tags": ", ".join(f"{kmf.vocab[j]} ({kmf.centers[c, j]:+.2f})" for j in under),
                     "most_popular_games": " | ".join(ds.titles[i] for i in np.argsort(-kmf.P[c])[:5]),
                     "distinctive_games": " | ".join(ds.titles[i] for i in np.argsort(-lift)[:5]),
                     "test_users": int(np.isin(test_users, members).sum())})
    pd.DataFrame(rows).to_csv(os.path.join(cfg.out_dir, "clusters.csv"), index=False)
    plots.cluster_heatmap(kmf.centers, kmf.vocab, sizes, os.path.join(fig_dir, "kmeans_cluster_tags.png"),
                          diverging=cfg.feature_scaling == "zscore")

    arf = comps["ar"]
    top_rules = [{"antecedent": " + ".join(ds.titles[i] for i in A), "consequent": ds.titles[b],
                  "support": round(sp, 4), "confidence": round(cf, 3), "lift": round(lf, 2)}
                 for A, b, sp, cf, lf in arf.rules[:100]]
    pd.DataFrame(top_rules).to_csv(os.path.join(cfg.out_dir, "apriori_top_rules.csv"), index=False)
    analysis = {"kmeans_generic_tags_removed": kmf.generic_tags,
                "kmeans_sse": float(kmf.sse), "kmeans_cluster_sizes": sizes.tolist(),
                "apriori_frequent_itemsets": len(arf.freq), "apriori_rules": len(arf.rules),
                "apriori_itemsets_by_size": {str(n): sum(1 for s in arf.freq if len(s) == n)
                                             for n in range(1, cfg.max_itemset_len + 1)},
                "apriori_test_user_rule_coverage": rule_coverage(arf, test_users, cfg.batch_size),
                "content_tag_vocab": int(len(comps["cb"].tag_vocab)),
                "val_candidate_recall": candidate_recall,
                "ablation_base": base_name, "test_users": int(len(test_users)),
                "python": platform.python_version(), "platform": platform.platform()}
    json.dump(analysis, open(os.path.join(cfg.out_dir, "component_analysis.json"), "w"), indent=2)

    # ------------------------------------------------------------------ figures
    main_models = {k: v for k, v in results.items() if "−" not in k}
    ablation = {k: v for k, v in results.items() if k == "Item-based CF" or k.startswith(base_name)}
    plots.metric_bars(main_models, tk, os.path.join(fig_dir, "test_ndcg_models.png"),
                      f"Test {tk.upper()}: baseline, each algorithm, hybrids")
    plots.metric_dots(ablation, tk, os.path.join(fig_dir, "test_ndcg_ablation.png"),
                      f"Ablation: removing one component (test {tk.upper()})")
    plots.group_recall(main_models, max(cfg.k_list), GROUP_NAMES,
                       os.path.join(fig_dir, "test_recall_by_popularity.png"))

    show = [c for c in res_df.columns if any(c == f"{m}@{k}" for m in ("precision", "recall", "ndcg", "coverage")
                                              for k in cfg.k_list)] + [tail_key]
    print("\n=== TEST RESULTS ===")
    print(res_df[show].astype(float).round(4).to_string())
    print(f"\nresults written to {cfg.out_dir}/")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    c = Config()
    p.add_argument("--data-dir", default=c.data_dir)
    p.add_argument("--out-dir", default=c.out_dir)
    p.add_argument("--seed", type=int, default=c.seed)
    p.add_argument("--max-users", type=int, default=c.max_users)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--min-user-pos", type=int, default=c.min_user_pos)
    p.add_argument("--min-item-pos", type=int, default=c.min_item_pos)
    p.add_argument("--split", choices=["temporal", "random"], default=c.split)
    p.add_argument("--feature-scaling", choices=["zscore", "share"], default=c.feature_scaling)
    p.add_argument("--n-candidates", type=int, default=c.n_candidates)
    p.add_argument("--no-description", action="store_true")
    p.add_argument("--supplement", default=c.supplement_path,
                   help="metadata supplement file from scripts/fetch_missing_tags.py")
    p.add_argument("--no-supplement", action="store_true", help="use Kaggle metadata only (v2 behaviour)")
    p.add_argument("--quick", action="store_true", help="small grids for a fast smoke test")
    a = p.parse_args()
    cfg = Config(data_dir=a.data_dir, out_dir=a.out_dir, seed=a.seed, max_users=a.max_users,
                 max_rows=a.max_rows, min_user_pos=a.min_user_pos, min_item_pos=a.min_item_pos,
                 split=a.split, feature_scaling=a.feature_scaling, n_candidates=a.n_candidates,
                 use_description=not a.no_description,
                 supplement_path=None if a.no_supplement else a.supplement)
    if a.quick:
        cfg.k_candidates, cfg.cf_neighbors_grid = (2, 4, 6), (50, 100)
        cfg.support_grid, cfg.kmeans_n_init = (0.005, 0.01), 2
        cfg.weight_grid = (0.0, 0.03, 0.1, 0.3)
    return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    main(parse())
