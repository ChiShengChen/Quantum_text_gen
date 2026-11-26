"""Check our BLEU / Self-BLEU against NLTK."""

import os
import random
import sys

from nltk.translate.bleu_score import corpus_bleu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qnlg import metrics as M  # noqa: E402

random.seed(0)
V = list("abcdefgh")


def rand_sent():
    # length >= max_n: for shorter hypotheses NLTK uses max(1, #ngrams) as the
    # precision denominator, whereas standard BLEU (multi-bleu.perl, sacreBLEU) uses 0
    return [random.choice(V) for _ in range(random.randint(4, 9))]


def test_paired_bleu():
    hyps = [rand_sent() for _ in range(60)]
    refs = [rand_sent() for _ in range(60)]
    ours = M.corpus_bleu_paired(hyps, refs, 4)
    for k in range(1, 5):
        w = tuple([1 / k] * k)
        ref = corpus_bleu([[r] for r in refs], hyps, weights=w)
        assert abs(ours[f"bleu{k}"] - ref) < 1e-9, (k, ours, ref)


def test_multiref_bleu():
    hyps = [rand_sent() for _ in range(40)]
    refs = [rand_sent() for _ in range(30)]
    ours = M.MultiRefBLEU(refs, 4).score(hyps)
    for k in range(1, 5):
        w = tuple([1 / k] * k)
        ref = corpus_bleu([refs] * len(hyps), hyps, weights=w)
        assert abs(ours[f"bleu{k}"] - ref) < 1e-9, (k, ours, ref)


def test_self_bleu():
    hyps = [rand_sent() for _ in range(35)] + [["a", "b", "c", "d"]] * 3
    ours = M.self_bleu(hyps, 4)
    refs = [hyps[:i] + hyps[i + 1:] for i in range(len(hyps))]
    ref = corpus_bleu(refs, hyps, weights=(0.25,) * 4)
    assert abs(ours - ref) < 1e-9, (ours, ref)


def test_degenerate_outputs_are_flagged():
    # the failure mode of the original submission: one prompt word per output
    hyps = [["the"], ["birds"], ["water"], ["spring"], ["stars"]]
    rep = M.generation_report(hyps, train_set=set())
    assert rep["distinct1"] == 1.0 and rep["consec_rep"] == 0.0  # looks "perfect" ...
    assert rep["single_token_rate"] == 1.0 and rep["mean_len"] == 1.0  # ... but is flagged


if __name__ == "__main__":
    test_paired_bleu()
    test_multiref_bleu()
    test_self_bleu()
    test_degenerate_outputs_are_flagged()
    print("all metric tests passed")
