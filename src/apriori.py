"""(2) A-Priori frequent itemsets + association rules, implemented from scratch.

Baskets = the set of games each user reviewed positively in the training part.
Support counting uses one bitset per item (bit u set if user u's basket holds the item);
the support of an itemset is popcount(AND of its item bitsets).
Candidates of size k are generated only from frequent (k-1)-itemsets that share a
(k-2)-prefix, and pruned unless every (k-1)-subset is frequent (monotonicity).
"""
from itertools import combinations
from math import ceil

import numpy as np
from scipy import sparse


def _popcount(x):
    return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")


def _item_bitsets(B):
    """B: binary csr (users x items) -> list of python-int bitsets, one per item."""
    n_users = B.shape[0]
    C = B.tocsc()
    out = []
    for j in range(B.shape[1]):
        col = np.zeros(n_users, dtype=bool)
        col[C.indices[C.indptr[j]:C.indptr[j + 1]]] = True
        out.append(int.from_bytes(np.packbits(col, bitorder="little").tobytes(), "little"))
    return out


def frequent_itemsets(B, min_support, max_len=3, max_candidates=None, log=None):
    """Returns dict {tuple(sorted items): support_count} and the number of baskets.
    max_candidates: safety cap - if a level would count more candidates than this,
    stop growing itemsets (reported through `log`)."""
    n = B.shape[0]
    min_count = max(1, ceil(min_support * n))
    counts = np.asarray(B.sum(axis=0)).ravel()
    bits = _item_bitsets(B)
    level = {(j,): bits[j] for j in np.flatnonzero(counts >= min_count)}   # L1
    freq = {k: _popcount(v) for k, v in level.items()}
    k = 2
    while level and k <= max_len:
        prev = sorted(level)
        prev_set = set(prev)
        if max_candidates:
            # upper bound on joins: sum over prefix groups of C(group size, 2)
            groups = {}
            for p in prev:
                groups[p[:-1]] = groups.get(p[:-1], 0) + 1
            bound = sum(g * (g - 1) // 2 for g in groups.values())
            if bound > max_candidates:
                if log:
                    log.warning("Apriori: level %d would join up to %d candidates (> %d); "
                                "stopping at itemsets of size %d", k, bound, max_candidates, k - 1)
                break
        nxt = {}
        for a in range(len(prev)):
            for b in range(a + 1, len(prev)):
                p, q = prev[a], prev[b]
                if p[:-1] != q[:-1]:          # sorted list -> later q's can't share prefix
                    break
                cand = p + (q[-1],)
                if any(s not in prev_set for s in combinations(cand, k - 1)):
                    continue                  # Apriori pruning
                bs = level[p] & bits[q[-1]]
                c = _popcount(bs)
                if c >= min_count:
                    nxt[cand] = bs
                    freq[cand] = c
        level = nxt
        k += 1
    return freq, n


def association_rules(freq, n, min_confidence, min_lift=1.0):
    """Single-consequent rules A -> b from every frequent itemset of size >= 2."""
    rules = []
    for itemset, c in freq.items():
        if len(itemset) < 2:
            continue
        for b in itemset:
            A = tuple(x for x in itemset if x != b)
            conf = c / freq[A]
            lift = conf / (freq[(b,)] / n)
            if conf >= min_confidence and lift >= min_lift:
                rules.append((A, b, c / n, conf, lift))
    rules.sort(key=lambda r: (-r[3], -r[4]))
    return rules


class RuleScorer:
    """Scores game b for user u as the max confidence of any rule A -> b whose
    antecedent A is fully contained in u's basket."""

    def __init__(self, rules, n_items):
        self.n_items = n_items
        self.rules = rules
        R = len(rules)
        if R == 0:
            self.A = None
            return
        rows, cols = [], []
        for r, (A, _, _, _, _) in enumerate(rules):
            rows += [r] * len(A)
            cols += list(A)
        self.A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                                   shape=(R, n_items))
        self.size = np.array([len(r[0]) for r in rules], np.float32)
        self.conf = np.array([r[3] for r in rules], np.float32)
        self.cons = np.array([r[1] for r in rules])
        self.groups = {b: np.flatnonzero(self.cons == b) for b in np.unique(self.cons)}

    def score(self, B_users):
        """B_users: binary csr (batch x items). Returns dense (batch x items)."""
        out = np.zeros((B_users.shape[0], self.n_items), np.float32)
        if self.A is None:
            return out
        # chunk users so the (users x rules) matrix stays around 50M cells
        step = max(1, int(5e7 // max(len(self.rules), 1)))
        for r0 in range(0, B_users.shape[0], step):
            hit = np.asarray((B_users[r0:r0 + step] @ self.A.T).todense()) >= self.size[None, :]
            fired = hit * self.conf[None, :]
            for b, idx in self.groups.items():
                out[r0:r0 + step, b] = fired[:, idx].max(axis=1)
        return out
