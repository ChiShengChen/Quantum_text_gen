"""Text-generation metrics.

BLEU follows Papineni et al. (2002): corpus-level, clipped n-gram precisions,
uniform weights, brevity penalty with the closest reference length.  With many
references per hypothesis (Test-BLEU, Self-BLEU) the clip count is the maximum
count of the n-gram in any single reference.  `tests/test_metrics.py` checks
this against NLTK's `corpus_bleu`.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict

import numpy as np


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_from_stats(match, total, hyp_len, ref_len, max_n):
    if hyp_len == 0:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        if total[n] == 0 or match[n] == 0:
            return 0.0
        logs.append(math.log(match[n] / total[n]))
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(sum(logs) / max_n)


def corpus_bleu_paired(hyps, refs, max_n=4):
    """One reference per hypothesis. hyps/refs: lists of token lists. Returns BLEU-1..max_n."""
    out = {}
    stats_m = defaultdict(int)
    stats_t = defaultdict(int)
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hc, rc = Counter(ngrams(h, n)), Counter(ngrams(r, n))
            stats_m[n] += sum(min(c, rc[g]) for g, c in hc.items())
            stats_t[n] += max(len(h) - n + 1, 0)
    for k in range(1, max_n + 1):
        out[f"bleu{k}"] = _bleu_from_stats(stats_m, stats_t, hyp_len, ref_len, k)
    return out


class MultiRefBLEU:
    """Corpus BLEU of hypotheses against one shared reference set (Texygen 'Test-BLEU')."""

    def __init__(self, refs, max_n=4):
        self.max_n = max_n
        self.maxc = [None] + [defaultdict(int) for _ in range(max_n)]
        for r in refs:
            for n in range(1, max_n + 1):
                for g, c in Counter(ngrams(r, n)).items():
                    if c > self.maxc[n][g]:
                        self.maxc[n][g] = c
        self.ref_lens = sorted({len(r) for r in refs})

    def _closest(self, L):
        i = bisect.bisect_left(self.ref_lens, L)
        cands = self.ref_lens[max(i - 1, 0): i + 1]
        return min(cands, key=lambda x: (abs(x - L), x))

    def score(self, hyps):
        m, t = defaultdict(int), defaultdict(int)
        hl = rl = 0
        for h in hyps:
            if not h:
                continue
            hl += len(h)
            rl += self._closest(len(h))
            for n in range(1, self.max_n + 1):
                for g, c in Counter(ngrams(h, n)).items():
                    m[n] += min(c, self.maxc[n].get(g, 0))
                t[n] += max(len(h) - n + 1, 0)
        return {f"bleu{k}": _bleu_from_stats(m, t, hl, rl, k) for k in range(1, self.max_n + 1)}


def self_bleu(hyps, max_n=4):
    """Self-BLEU (Zhu et al., 2018): each hypothesis scored against all *other* hypotheses.

    Computed exactly in O(N) by keeping, for every n-gram, the largest and second
    largest per-sentence count and how many sentences attain the maximum.
    """
    hyps = [h for h in hyps if h]
    if len(hyps) < 2:
        return 0.0
    top1 = [None] + [defaultdict(int) for _ in range(max_n)]
    top1_n = [None] + [defaultdict(int) for _ in range(max_n)]
    top2 = [None] + [defaultdict(int) for _ in range(max_n)]
    counts = []
    for h in hyps:
        per = [None]
        for n in range(1, max_n + 1):
            c = Counter(ngrams(h, n))
            per.append(c)
            for g, k in c.items():
                if k > top1[n][g]:
                    top2[n][g] = top1[n][g]
                    top1[n][g], top1_n[n][g] = k, 1
                elif k == top1[n][g]:
                    top1_n[n][g] += 1
                elif k > top2[n][g]:
                    top2[n][g] = k
        counts.append(per)
    # length multiset for closest-other-reference brevity penalty
    len_count = Counter(len(h) for h in hyps)
    lens_sorted = sorted(len_count)
    m, t = defaultdict(int), defaultdict(int)
    hl = rl = 0
    for h, per in zip(hyps, counts):
        L = len(h)
        hl += L
        avail = [x for x in lens_sorted if x != L or len_count[x] > 1]
        i = bisect.bisect_left(avail, L)
        cands = avail[max(i - 1, 0): i + 1]
        rl += min(cands, key=lambda x: (abs(x - L), x))
        for n in range(1, max_n + 1):
            for g, k in per[n].items():
                if k == top1[n][g] and top1_n[n][g] == 1:
                    other = top2[n][g]
                else:
                    other = top1[n][g]
                m[n] += min(k, other)
            t[n] += max(L - n + 1, 0)
    return _bleu_from_stats(m, t, hl, rl, max_n)


def distinct_n(hyps, n):
    grams = [g for h in hyps for g in ngrams(h, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_stats(hyps):
    """consecutive repetition rate (micro), and seq-rep-2 (Welleck et al., 2020)."""
    pairs = reps = 0
    seq_rep = []
    for h in hyps:
        pairs += max(len(h) - 1, 0)
        reps += sum(h[i] == h[i + 1] for i in range(len(h) - 1))
        bg = ngrams(h, 2)
        if bg:
            seq_rep.append(1 - len(set(bg)) / len(bg))
    return {"consec_rep": reps / pairs if pairs else 0.0,
            "seq_rep2": float(np.mean(seq_rep)) if seq_rep else 0.0}


def generation_report(hyps, train_set, test_refs=None, unk="<unk>"):
    lens = [len(h) for h in hyps]
    rep = {
        "n": len(hyps),
        "mean_len": float(np.mean(lens)) if lens else 0.0,
        "empty_rate": float(np.mean([l == 0 for l in lens])),
        "single_token_rate": float(np.mean([l == 1 for l in lens])),
        "unk_rate": (sum(h.count(unk) for h in hyps) / max(sum(lens), 1)),
        "distinct1": distinct_n(hyps, 1),
        "distinct2": distinct_n(hyps, 2),
        "novelty": float(np.mean([" ".join(h) not in train_set for h in hyps])) if hyps else 0.0,
        "unique_rate": len({" ".join(h) for h in hyps}) / max(len(hyps), 1),
        "self_bleu4": self_bleu(hyps, 4),
    }
    rep.update(repetition_stats(hyps))
    if test_refs is not None:
        tb = MultiRefBLEU(test_refs, 4).score(hyps)
        rep.update({f"test_{k}": v for k, v in tb.items()})
    return rep
