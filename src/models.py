"""Recommenders. Every model exposes  fit(...)  and  score(users) -> dense (batch x items).

  Popularity      baseline: number of positive reviews in the fitting data
  ClusterPop      (1) K-means:  popularity inside the user's taste cluster
  RuleRec         (2) Apriori:  max confidence of fired association rules
  ItemCF          (3) item-based collaborative filtering (cosine, top-N neighbours)
  ContentBased    (4) TF-IDF tag/description profile, cosine similarity
  RerankHybrid    v2: CF proposes top-N candidates; K-means, Apriori and content
                  scores re-rank only those candidates
"""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import kmeans as km
from .apriori import frequent_itemsets, association_rules, RuleScorer


def _binary(X):
    B = X.copy().tocsr()
    B.data[:] = 1.0
    return B


class Popularity:
    name = "Popularity"

    def fit(self, X):
        self.pop = np.asarray(_binary(X).sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class ClusterPop:
    """(1) K-means role: group users by taste, recommend what their group likes."""
    name = "K-means (cluster popularity)"

    def __init__(self, k, n_tags, n_init, max_iter, seed, generic_frac=0.3, scaling="zscore"):
        self.k, self.n_tags, self.n_init, self.max_iter, self.seed = k, n_tags, n_init, max_iter, seed
        self.generic_frac, self.scaling = generic_frac, scaling

    def fit(self, X, item_tags):
        self.vocab, self.generic_tags = km.top_tags(item_tags, self.n_tags, self.generic_frac)
        self.T = km.item_tag_matrix(item_tags, self.vocab)
        self.F, self.feat_stats = km.user_features(X, self.T, self.scaling)
        self.labels, self.centers, self.sse = km.kmeans(
            self.F, self.k, self.n_init, self.max_iter, seed=self.seed)
        B = _binary(X)
        P = np.zeros((self.k, X.shape[1]), np.float32)
        for c in range(self.k):
            members = np.flatnonzero(self.labels == c)
            if len(members):
                P[c] = np.asarray(B[members].mean(axis=0)).ravel()
        self.P = P
        self.global_pop = np.asarray(B.mean(axis=0)).ravel()
        return self

    def score(self, users, tie_break=True):
        # tiny global-popularity term breaks ties among games nobody in the cluster liked
        s = self.P[self.labels[users]]
        return s + 1e-6 * self.global_pop[None, :] if tie_break else s


class RuleRec:
    """(2) Apriori role: 'players who liked {A} also liked b' rules."""
    name = "Apriori (association rules)"

    def __init__(self, min_support, min_confidence, min_lift, max_len, max_candidates=None, log=None):
        self.min_support, self.min_confidence = min_support, min_confidence
        self.min_lift, self.max_len = min_lift, max_len
        self.max_candidates, self.log = max_candidates, log

    def fit(self, X):
        self.B = _binary(X)
        self.freq, self.n = frequent_itemsets(self.B, self.min_support, self.max_len,
                                              self.max_candidates, self.log)
        self.rules = association_rules(self.freq, self.n, self.min_confidence, self.min_lift)
        self.scorer = RuleScorer(self.rules, X.shape[1])
        self.pop = np.asarray(self.B.mean(axis=0)).ravel()
        return self

    def score(self, users, tie_break=True):
        s = self.scorer.score(self.B[users])
        return s + 1e-6 * self.pop[None, :] if tie_break else s


class ItemCF:
    """(3) Item-based CF: s(u, j) = sum_i x_ui * sim(i, j), sim = cosine over users,
    keeping only each target game's top-N most similar games."""
    name = "Item-based CF"

    def __init__(self, n_neighbors=100, chunk=2000):
        self.n_neighbors, self.chunk = n_neighbors, chunk

    def fit(self, X):
        self.X = X.tocsr()
        Xn = normalize(self.X, axis=0)                      # column-wise L2
        S = (Xn.T @ Xn).tocsr()
        n = S.shape[0]
        k = min(self.n_neighbors, n - 1)
        rows, cols, vals = [], [], []
        for r0 in range(0, n, self.chunk):
            D = S[r0:r0 + self.chunk].toarray()
            D[np.arange(D.shape[0]), np.arange(r0, r0 + D.shape[0])] = 0.0   # no self-similarity
            top = np.argpartition(-D, k - 1, axis=1)[:, :k]
            v = np.take_along_axis(D, top, axis=1)
            keep = v > 0
            rr = np.repeat(np.arange(r0, r0 + D.shape[0]), k).reshape(-1, k)
            rows.append(rr[keep]); cols.append(top[keep]); vals.append(v[keep])
        # row j = neighbours of target j  ->  score = X_u @ W.T
        W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, n), dtype=np.float32)
        self.WT = W.T.tocsr()
        return self

    def score(self, users):
        return (self.X[users] @ self.WT).toarray()


class ContentBased:
    """(4) Content-based filtering: TF-IDF over tags (+ description); the user profile
    is the playtime-weighted mean of liked games' vectors."""
    name = "Content-based"

    def __init__(self, use_description=True, desc_weight=0.5, max_desc_features=5000):
        self.use_description, self.desc_weight = use_description, desc_weight
        self.max_desc_features = max_desc_features

    def build_item_vectors(self, item_tags, item_desc):
        # each whole tag ("Open World") is one token
        tag_vec = TfidfVectorizer(analyzer=lambda tags: tags, min_df=1)
        blocks = [normalize(tag_vec.fit_transform(item_tags))]
        self.tag_vocab = tag_vec.get_feature_names_out()
        if self.use_description and any(d.strip() for d in item_desc):
            dv = TfidfVectorizer(stop_words="english", max_features=self.max_desc_features,
                                 min_df=2, sublinear_tf=True)
            try:
                blocks.append(self.desc_weight * normalize(dv.fit_transform(item_desc)))
            except ValueError:            # vocabulary empty after min_df
                pass
        self.V = normalize(sparse.hstack(blocks).tocsr())
        return self

    def fit(self, X, item_tags, item_desc):
        self.build_item_vectors(item_tags, item_desc)
        self.X = X.tocsr()
        return self

    def score(self, users):
        prof = normalize(self.X[users] @ self.V)
        return (prof @ self.V.T).toarray()


def minmax_rows(S):
    lo = S.min(axis=1, keepdims=True)
    hi = S.max(axis=1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-12)


class RerankHybrid:
    """Two-stage hybrid that follows the design diagram:

      stage 1  Item-based CF proposes each user's top-N unseen games (candidates)
      stage 2  final(u, j) = CF(u, j) + w_km*KM(u, j) + w_ar*AR(u, j)
                             + w_cb*CB(u, j)*(1 + beta*cold_j)          for j in candidates

    Every component is min-max normalised over the user's N candidates, so a small
    weight can only reorder games CF already considers plausible. cold_j = 1 for games
    below the cold popularity quantile (content compensates for thin history).
    With all weights 0 the result is exactly CF's ranking."""
    name = "Hybrid (re-rank)"
    keys = ("km", "ar", "cb")

    def __init__(self, cf, comps, seen, n_candidates, cold, cold_boost, weights=None):
        self.cf, self.comps, self.seen = cf, comps, seen.tocsr()
        self.n_candidates = n_candidates
        self.cold, self.cold_boost = cold.astype(np.float32), cold_boost
        self.weights = weights or {k: 0.0 for k in self.keys}

    def prepare(self, users):
        """Candidates and normalised component scores (batch x N)."""
        S = self.cf.score(users).astype(np.float32)
        r, c = self.seen[users].nonzero()
        S[r, c] = -np.inf
        N = min(self.n_candidates, S.shape[1] - 1)
        cand = np.argpartition(-S, N - 1, axis=1)[:, :N]
        cf = np.take_along_axis(S, cand, axis=1)
        cf[~np.isfinite(cf)] = np.nan                      # fewer than N unseen games
        valid = ~np.isnan(cf)
        out = {"cand": cand, "valid": valid, "cf": minmax_rows(np.nan_to_num(cf, nan=0.0))}
        for key in self.keys:
            m = self.comps[key]
            s = m.score(users, tie_break=False) if isinstance(m, (ClusterPop, RuleRec)) \
                else m.score(users)
            s = np.take_along_axis(s.astype(np.float32), cand, axis=1)
            if key == "cb":
                s = s * (1.0 + self.cold_boost * self.cold[cand])
            out[key] = minmax_rows(s)
        return out

    @staticmethod
    def combine(prep, weights):
        total = prep["cf"].copy()
        for key in RerankHybrid.keys:
            w = weights.get(key, 0.0)
            if w:
                total += w * prep[key]
        total[~prep["valid"]] = -np.inf
        return total

    @staticmethod
    def ranked_items(prep, weights, K):
        total = RerankHybrid.combine(prep, weights)
        order = np.argsort(-total, axis=1, kind="stable")[:, :K]
        return np.take_along_axis(prep["cand"], order, axis=1)

    def score(self, users):
        """Dense scores for the generic evaluator: candidates keep their hybrid score,
        everything else is pushed below them."""
        prep = self.prepare(users)
        total = self.combine(prep, self.weights)
        out = np.full((len(users), self.seen.shape[1]), -1e9, np.float32)
        np.put_along_axis(out, prep["cand"], np.where(np.isfinite(total), total, -1e9), axis=1)
        return out
