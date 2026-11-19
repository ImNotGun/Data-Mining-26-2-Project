"""Top-K ranking evaluation with a shared protocol for every model:
same users, same candidate set (all games minus the user's already-seen games),
same held-out positive targets.

RankAccumulator holds the metric definitions; evaluate() feeds it from dense score
matrices, and the hybrid weight search feeds it ranked lists directly (same maths)."""
import numpy as np


def topk_from_scores(S, seen_rows, K):
    S = S.astype(np.float32, copy=True)
    r, c = seen_rows.nonzero()
    S[r, c] = -np.inf
    top = np.argpartition(-S, K - 1, axis=1)[:, :K]
    order = np.argsort(-np.take_along_axis(S, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1)


class RankAccumulator:
    def __init__(self, k_list, n_items, item_group=None):
        self.k_list, self.Kmax = tuple(k_list), max(k_list)
        self.item_group = item_group
        self.per = {f"{m}@{k}": [] for k in self.k_list for m in ("precision", "recall", "ndcg", "hit")}
        self.rec_items = {k: np.zeros(n_items, bool) for k in self.k_list}
        self.g_hits, self.g_tot = {}, {}
        self.disc = 1.0 / np.log2(np.arange(2, self.Kmax + 2))

    def add(self, top, T):
        """top: (batch x >=Kmax) ranked item ids; T: csr targets for the same users."""
        disc = self.disc
        for row, u_top in enumerate(top):
            t = T.indices[T.indptr[row]:T.indptr[row + 1]]
            if len(t) == 0:
                continue
            rel = np.isin(u_top, t)
            for k in self.k_list:
                h = rel[:k].sum()
                self.per[f"precision@{k}"].append(h / k)
                self.per[f"recall@{k}"].append(h / len(t))
                self.per[f"hit@{k}"].append(float(h > 0))
                idcg = disc[:min(len(t), k)].sum()
                self.per[f"ndcg@{k}"].append((disc[:k] * rel[:k]).sum() / idcg)
                self.rec_items[k][u_top[:k]] = True
            if self.item_group is not None:
                hit_items = set(u_top[:self.Kmax][rel[:self.Kmax]])
                for it in t:
                    g = int(self.item_group[it])
                    self.g_tot[g] = self.g_tot.get(g, 0) + 1
                    self.g_hits[g] = self.g_hits.get(g, 0) + (it in hit_items)

    def result(self):
        summary = {}
        for key, v in self.per.items():
            v = np.asarray(v)
            summary[key] = v.mean() if len(v) else np.nan
            summary[key + "_ci95"] = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        for k in self.k_list:
            summary[f"coverage@{k}"] = self.rec_items[k].mean()
        for g in sorted(self.g_tot):
            summary[f"recall@{self.Kmax}_group{g}"] = self.g_hits[g] / self.g_tot[g]
        summary["n_users"] = len(self.per[f"recall@{self.k_list[0]}"])
        return summary, self.per


def evaluate(score_fn, users, targets, seen, k_list, batch_size=1000, item_group=None):
    """score_fn(users) -> dense scores. targets/seen: csr (all users x items).
    item_group: optional int array (items,) with popularity group id per game.
    Returns (summary dict, per-user metric lists)."""
    acc = RankAccumulator(k_list, targets.shape[1], item_group)
    for b0 in range(0, len(users), batch_size):
        ub = users[b0:b0 + batch_size]
        acc.add(topk_from_scores(score_fn(ub), seen[ub], max(k_list)), targets[ub])
    return acc.result()
