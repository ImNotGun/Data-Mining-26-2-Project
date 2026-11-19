"""Experiment configuration (v2). Every tunable value lives here so a run is fully
described by the config.json saved next to its results.

v2 changes vs. v1 (see README "v1 -> v2"):
  K-means : generic tags removed, tag shares z-scored, k grid starts at 2
  Apriori : lower support grid (more rules), lower min confidence
  CF      : neighbour grid extended to 400
  Hybrid  : CF generates top-N candidates, other components only RE-RANK them;
            finer weight grid with small weights; an extra tail-aware selection
"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import json


@dataclass
class Config:
    # ---- paths / reproducibility ----
    data_dir: str = "data/raw"
    out_dir: str = "results"
    seed: int = 42

    # ---- subset construction (Kaggle data has ~41M reviews) ----
    max_rows: Optional[int] = None      # read only the first N rows (debugging)
    min_user_pos: int = 10              # k-core: user needs >= N positive reviews
    min_item_pos: int = 30              # k-core: game needs >= N positive users
    max_users: int = 30000              # uniform random sample of users after k-core

    # ---- metadata supplement (v3): fills games whose Kaggle tags/description are empty ----
    supplement_path: Optional[str] = "data/supplement/supplement_metadata.jsonl"

    # ---- split ----
    split: str = "temporal"             # "temporal" (by review date) or "random"
    test_ratio: float = 0.2
    val_ratio: float = 0.2              # fraction of the remaining (non-test) part

    # ---- (1) K-means ----
    n_tags: int = 60                    # tag vocabulary size for user features
    generic_tag_frac: float = 0.3       # drop tags attached to > 30% of games (e.g. Singleplayer)
    feature_scaling: str = "zscore"     # "zscore" (v2) or "share" (v1: raw playtime shares)
    k_candidates: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)
    kmeans_n_init: int = 5
    kmeans_max_iter: int = 100
    silhouette_sample: int = 3000

    # ---- (2) Apriori ----
    support_grid: Tuple[float, ...] = (0.003, 0.005, 0.01)
    min_confidence: float = 0.1
    min_lift: float = 1.0
    max_itemset_len: int = 3
    max_candidates: int = 3_000_000     # safety cap per level (stops growing itemsets)

    # ---- (3) Item-based CF ----
    cf_neighbors_grid: Tuple[int, ...] = (50, 100, 200, 400)

    # ---- (4) Content-based ----
    use_description: bool = True
    desc_weight: float = 0.5            # description block weight vs. tag block (1.0)
    max_desc_features: int = 5000

    # ---- hybrid (CF candidates -> re-rank) ----
    n_candidates: int = 100             # CF top-N candidates that get re-ranked
    weight_grid: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
    cold_boost: float = 1.0             # extra content weight for low-history games
    cold_quantile: float = 0.5          # games below this popularity quantile are "cold"
    tail_tolerance: float = 0.99        # tail-aware pick: best tail recall with NDCG >= 99% of best

    # ---- evaluation ----
    k_list: Tuple[int, ...] = (10, 20)
    tune_k: int = 20                    # model selection uses NDCG@tune_k on validation
    tune_users: int = 3000              # validation users sampled for tuning
    batch_size: int = 1000

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
