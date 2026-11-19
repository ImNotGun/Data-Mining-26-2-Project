"""Generate a small synthetic dataset with the SAME schema as the Kaggle
"Game Recommendations on Steam" files. Used only to smoke-test the pipeline;
never report results from it.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main(out, n_users=4000, n_games=800, seed=0):
    rng = np.random.default_rng(seed)
    genres = {
        "Action": ["Action", "Shooter", "FPS", "Fast-Paced"],
        "RPG": ["RPG", "Open World", "Story Rich", "Fantasy"],
        "Strategy": ["Strategy", "Turn-Based", "Simulation", "Management"],
        "Indie": ["Indie", "Pixel Graphics", "Platformer", "Roguelike"],
        "Horror": ["Horror", "Survival", "Atmospheric", "Dark"],
        "Casual": ["Casual", "Puzzle", "Relaxing", "Cute"],
    }
    gnames = list(genres)
    extra = ["Multiplayer", "Singleplayer", "Co-op", "Great Soundtrack", "Difficult", "Early Access"]
    words = {g: [w.lower() + s for w in genres[g] for s in ("", "s", "ing")] for g in gnames}

    app_ids = rng.choice(np.arange(10, 2_000_000), n_games, replace=False)
    g_main = rng.integers(len(gnames), size=n_games)
    pop = rng.pareto(1.2, n_games) + 0.05
    games, meta = [], []
    for j, a in enumerate(app_ids):
        g = gnames[g_main[j]]
        tags = list(rng.choice(genres[g], 3, replace=False))
        if rng.random() < 0.4:
            g2 = gnames[rng.integers(len(gnames))]
            tags.append(rng.choice(genres[g2]))
        tags += list(rng.choice(extra, 2, replace=False))
        desc = " ".join(rng.choice(words[g], 12)) + " game with " + " ".join(rng.choice(words[g], 4))
        games.append({"app_id": int(a), "title": f"{g} Game {j}", "date_release": "2020-01-01",
                      "win": True, "mac": False, "linux": False, "rating": "Positive",
                      "positive_ratio": int(rng.integers(40, 100)), "user_reviews": 0,
                      "price_final": round(float(rng.uniform(0, 60)), 2),
                      "price_original": 0.0, "discount": 0.0, "steam_deck": True})
        if rng.random() < 0.03:          # some games have no metadata row (as in the real data)
            continue
        meta.append({"app_id": int(a), "description": desc if rng.random() > 0.05 else "",
                     "tags": [str(t) for t in dict.fromkeys(tags)]})

    rows, rid = [], 0
    start = np.datetime64("2019-01-01")
    for u in range(n_users):
        pref = rng.dirichlet(np.full(len(gnames), 0.3))
        n = int(np.clip(rng.lognormal(2.6, 0.6), 3, 120))
        w = pop * (0.05 + pref[g_main])
        items = rng.choice(n_games, min(n, n_games), replace=False, p=w / w.sum())
        days = np.sort(rng.integers(0, 1500, len(items)))
        uid = int(rng.integers(1e6, 1e9))
        for j, d in zip(items, days):
            like = pref[g_main[j]] * len(gnames)
            rec = rng.random() < 1 / (1 + np.exp(-(like - 0.3)))
            hours = float(np.round(rng.lognormal(2.5 + 0.8 * rec, 1.0), 1))
            rows.append({"app_id": int(app_ids[j]), "helpful": 0, "funny": 0,
                         "date": str(start + np.timedelta64(int(d), "D")),
                         "is_recommended": bool(rec), "hours": hours, "user_id": uid,
                         "review_id": rid})
            rid += 1
    rec = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    rec.to_csv(os.path.join(out, "recommendations.csv"), index=False)
    pd.DataFrame(games).to_csv(os.path.join(out, "games.csv"), index=False)
    rec.groupby("user_id").agg(products=("app_id", "size"), reviews=("review_id", "size")) \
       .reset_index().to_csv(os.path.join(out, "users.csv"), index=False)
    with open(os.path.join(out, "games_metadata.json"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"wrote {len(rec)} reviews, {n_users} users, {n_games} games to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--users", type=int, default=4000)
    p.add_argument("--games", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.out, a.users, a.games, a.seed)
