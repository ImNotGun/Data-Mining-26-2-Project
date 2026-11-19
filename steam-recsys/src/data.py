"""Turns the preprocessed review list (src/preprocess.py) into index-based tables and
sparse matrices for the models, plus descriptive statistics.

Files expected in data_dir:
  recommendations.csv  app_id, helpful, funny, date, is_recommended, hours, user_id, review_id
  games.csv            app_id, title, date_release, ..., positive_ratio, user_reviews, price_final, ...
  games_metadata.json  one JSON object per line: {"app_id", "description", "tags": [...]}
"""
import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ hand-off
# Loading, filtering and splitting live in src/preprocess.py (plain Python, no pandas).
# Below, the finished review list is turned into tables / sparse matrices for the models.
def build_dataset_from_reviews(reviews, titles, meta):
    df = pd.DataFrame(reviews, columns=["user_id", "app_id", "is_recommended", "hours", "date", "part"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    games = pd.DataFrame({"app_id": list(titles.keys()), "title": list(titles.values())})
    meta_df = pd.DataFrame([{"app_id": a, "description": m["description"], "tags": m["tags"]}
                            for a, m in meta.items()], columns=["app_id", "description", "tags"])
    return build_dataset(df, games, meta_df)


# ---------------------------------------------------------------------- matrices
@dataclass
class Dataset:
    user_ids: np.ndarray        # index -> original user_id
    item_ids: np.ndarray        # index -> app_id
    df: pd.DataFrame            # interactions with u, i, part columns
    titles: np.ndarray          # title per item index
    item_tags: list             # list of tag lists per item index
    item_desc: list             # description per item index
    item_has_meta: np.ndarray   # True if games_metadata.json has a row for the game

    @property
    def n_users(self):
        return len(self.user_ids)

    @property
    def n_items(self):
        return len(self.item_ids)

    def confidence(self, parts):
        """User x item matrix of positive reviews in `parts`; value = 1 + log1p(hours)."""
        d = self.df[self.df.part.isin(parts) & self.df.is_recommended]
        vals = 1.0 + np.log1p(d.hours.clip(lower=0).fillna(0).to_numpy(np.float32))
        return sparse.csr_matrix((vals, (d.u, d.i)), shape=(self.n_users, self.n_items),
                                 dtype=np.float32)

    def seen(self, parts):
        """All reviewed games (positive or negative) in `parts` -> masked at ranking time."""
        d = self.df[self.df.part.isin(parts)]
        return sparse.csr_matrix((np.ones(len(d), np.float32), (d.u, d.i)),
                                 shape=(self.n_users, self.n_items))

    def targets(self, part):
        d = self.df[(self.df.part == part) & self.df.is_recommended]
        return sparse.csr_matrix((np.ones(len(d), np.float32), (d.u, d.i)),
                                 shape=(self.n_users, self.n_items))


def build_dataset(split_df, games, meta):
    user_ids = np.sort(split_df.user_id.unique())
    item_ids = np.sort(split_df.app_id.unique())
    df = split_df.copy()
    df["u"] = np.searchsorted(user_ids, df.user_id)
    df["i"] = np.searchsorted(item_ids, df.app_id)
    g = games.set_index("app_id")
    titles = g.reindex(item_ids)["title"].fillna("").astype(str).to_numpy() \
        if "title" in g.columns else np.array([str(a) for a in item_ids])
    m = meta.set_index("app_id").reindex(item_ids)
    has_meta = m["tags"].notna().to_numpy()
    tags = [t if isinstance(t, list) else [] for t in m["tags"]]
    desc = m["description"].fillna("").tolist()
    return Dataset(user_ids, item_ids, df, titles, tags, desc, has_meta)


def missing_metadata_report(ds):
    """Games without tags or description: is the metadata row missing, or empty?"""
    pos = ds.df[ds.df.is_recommended].groupby("i").size()
    rows = []
    for i in range(ds.n_items):
        no_tags, no_desc = not ds.item_tags[i], not ds.item_desc[i].strip()
        if no_tags or no_desc:
            rows.append({"app_id": int(ds.item_ids[i]), "title": ds.titles[i],
                         "metadata_row_exists": bool(ds.item_has_meta[i]),
                         "no_tags": no_tags, "no_description": no_desc,
                         "positive_reviews": int(pos.get(i, 0))})
    return pd.DataFrame(rows, columns=["app_id", "title", "metadata_row_exists", "no_tags",
                                       "no_description", "positive_reviews"]) \
             .sort_values("positive_reviews", ascending=False)


def describe(ds):
    """Descriptive statistics for the report's data section."""
    df = ds.df
    pos = df[df.is_recommended]
    per_user = pos.groupby("u").size()
    per_item = pos.groupby("i").size()
    n_tags = sum(1 for t in ds.item_tags if t)
    stats = {
        "users": ds.n_users,
        "games": ds.n_items,
        "reviews": len(df),
        "positive_reviews": len(pos),
        "positive_ratio": round(len(pos) / max(len(df), 1), 4),
        "density": round(len(pos) / (ds.n_users * ds.n_items), 6),
        "pos_per_user_median": float(per_user.median()),
        "pos_per_user_mean": round(float(per_user.mean()), 2),
        "pos_per_game_median": float(per_item.median()),
        "hours_median": float(df.hours.median()),
        "games_with_tags": n_tags,
        "games_with_description": sum(1 for d in ds.item_desc if d.strip()),
    }
    for p in ["train", "val", "test"]:
        stats[f"{p}_reviews"] = int((df.part == p).sum())
        stats[f"{p}_positive"] = int(((df.part == p) & df.is_recommended).sum())
    if "date" in df.columns:
        stats["date_min"] = str(df.date.min().date()) if df.date.notna().any() else ""
        stats["date_max"] = str(df.date.max().date()) if df.date.notna().any() else ""
    return stats
