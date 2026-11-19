"""(1) K-means clustering of users by playtime-weighted tag (genre) shares.
Implemented from scratch (k-means++ initialisation + Lloyd iterations)."""
import numpy as np
from scipy import sparse
from sklearn.metrics import silhouette_score  # evaluation only


def top_tags(item_tags, n_tags, generic_frac=1.0):
    """Most frequent tags, skipping 'generic' tags attached to more than
    generic_frac of all games (they say little about taste)."""
    counts = {}
    for tags in item_tags:
        for t in set(tags):
            counts[t] = counts.get(t, 0) + 1
    limit = generic_frac * len(item_tags)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    generic = [t for t, c in ranked if c > limit]
    vocab = [t for t, c in ranked if c <= limit][:n_tags]
    return vocab, generic


def item_tag_matrix(item_tags, vocab):
    idx = {t: j for j, t in enumerate(vocab)}
    rows, cols = [], []
    for i, tags in enumerate(item_tags):
        for t in tags:
            if t in idx:
                rows.append(i)
                cols.append(idx[t])
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                             shape=(len(item_tags), len(vocab)))


def user_features(X_conf, T, scaling="zscore", stats=None):
    """Row u = share of user u's (log-)playtime spent on each tag (rows sum to 1).
    scaling="zscore": each tag column is standardised (mean 0, sd 1 over users), so a
    cluster is defined by tags a group plays MORE/LESS than average, not by tags
    everyone plays. Returns (F, stats) - stats = (mean, sd) to reuse on other users."""
    F = np.asarray((X_conf @ T).todense(), dtype=np.float64)
    s = F.sum(axis=1, keepdims=True)
    empty = s[:, 0] == 0
    F[~empty] /= s[~empty]
    if empty.any():                                  # no tagged games -> global mean profile
        F[empty] = F[~empty].mean(axis=0) if (~empty).any() else 1.0 / F.shape[1]
    if scaling == "zscore":
        if stats is None:
            stats = (F.mean(0), F.std(0) + 1e-9)
        F = (F - stats[0]) / stats[1]
    return F, stats


def _kmeans_pp(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        c = X[rng.choice(n, p=p)]
        centers.append(c)
        d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
    return np.array(centers)


def _assign(X, C):
    # squared euclidean distance via ||x||^2 - 2xc + ||c||^2
    d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    lab = d.argmin(1)
    return lab, np.maximum(d[np.arange(len(X)), lab], 0).sum()


def kmeans(X, k, n_init=5, max_iter=100, tol=1e-6, seed=0):
    """Returns (labels, centers, sse) of the best of n_init runs."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = _kmeans_pp(X, k, rng)
        prev = np.inf
        for _ in range(max_iter):
            lab, sse = _assign(X, C)
            for j in range(k):
                members = X[lab == j]
                C[j] = members.mean(0) if len(members) else X[rng.integers(len(X))]
            if prev - sse <= tol * max(prev, 1e-12):
                break
            prev = sse
        lab, sse = _assign(X, C)
        if best is None or sse < best[2]:
            best = (lab, C.copy(), sse)
    return best


def select_k(X, k_candidates, n_init, max_iter, sample, seed):
    """Elbow (SSE) + silhouette on a fixed user sample; returns table and best k."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    rows = []
    for k in k_candidates:
        lab, _, sse = kmeans(X, k, n_init, max_iter, seed=seed)
        sil = silhouette_score(X[idx], lab[idx]) if len(set(lab[idx])) > 1 else np.nan
        rows.append({"k": k, "sse": sse, "silhouette": sil})
    best_k = max(rows, key=lambda r: -np.inf if np.isnan(r["silhouette"]) else r["silhouette"])["k"]
    return rows, best_k
