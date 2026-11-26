"""Evaluation-only studies on trained checkpoints (no retraining).

  shots: re-evaluate hybrid models with finite-shot measurement estimates
      python -m qnlg.evaluate shots --dataset haiku --model qasa --seed 0 --shots 64 256 1024 4096
  temps: free generation at several sampling temperatures (pure sampling, no nucleus truncation)
      python -m qnlg.evaluate temps --dataset haiku --model qasa --seed 0 --temps 0.6 0.8 1.0 1.2 1.4

Checkpoints are read from experiments_rev/main/<dataset>/<model>/seed<k>.pt.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import metrics as M
from . import quantum as Q
from .data import Vocab, load_corpus, make_loader, map_unk
from .models import Config, build_model
from .train import evaluate_tokens, generate, set_seed


def load(dataset, model_name, seed, root, device):
    ck = torch.load(os.path.join(root, dataset, model_name, f"seed{seed}.pt"), map_location=device, weights_only=True)
    cfg = Config(**ck["config"])
    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    splits, _ = load_corpus(dataset)
    vocab = Vocab(splits["train"], min_freq=1 if dataset == "ptb" else 2)
    assert len(vocab) == cfg.vocab_size
    return model, cfg, splits, vocab


def prompted_bleu(model, splits, vocab, max_len, n_prompts=1000, P=2):
    test = [map_unk(s, vocab) for s in splits["test"]]
    cand = [s for s in test if len(s) > P][:n_prompts]
    prefix = torch.tensor([[vocab.bos] + [vocab.stoi[w] for w in s[:P]] for s in cand])
    conts = generate(model, prefix, vocab, max_new=max_len - P - 1, max_len=max_len, greedy=True)
    return M.corpus_bleu_paired(conts, [s[P:] for s in cand], 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["shots", "temps"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="experiments_rev/main")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--temps", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4])
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    out_root = a.out or f"experiments_rev/{a.mode}"
    out_path = os.path.join(out_root, a.dataset, a.model, f"seed{a.seed}.json")
    if os.path.exists(out_path):
        print("exists, skipping", out_path)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = torch.device(a.device)
    if not os.path.exists(os.path.join(a.root, a.dataset, a.model, f"seed{a.seed}.pt")):
        print("no checkpoint, skipping", a.dataset, a.model, a.seed)
        return
    model, cfg, splits, vocab = load(a.dataset, a.model, a.seed, a.root, device)
    te = make_loader(splits["test"], vocab, cfg.max_len, 256, False)
    res = {"args": vars(a), "results": []}

    if a.mode == "shots":
        for S in [None] + sorted(a.shots):
            Q.SHOTS = S
            set_seed(1000 + a.seed)  # same noise stream for every model
            tok = evaluate_tokens(model, te, vocab.pad, device)
            bl = prompted_bleu(model, splits, vocab, cfg.max_len)
            res["results"].append({"shots": S, "test": tok, "prompted": bl})
            print(f"shots={S}: ppl {tok['ppl']:.3f} acc {tok['acc']:.3f} BLEU-2 {bl['bleu2']:.4f}", flush=True)
        Q.SHOTS = None
    else:
        test_sents = [map_unk(s, vocab) for s in splits["test"]]
        train_set = {" ".join(map_unk(s, vocab)) for s in splits["train"]}
        for T in a.temps:
            g = torch.Generator(device=device)
            g.manual_seed(a.seed + 777)
            prefix = torch.full((a.n_samples, 1), vocab.bos, dtype=torch.long)
            samples = generate(model, prefix, vocab, max_new=cfg.max_len - 1, max_len=cfg.max_len, greedy=False,
                               top_p=1.0, temperature=T, gen=g)
            rep = M.generation_report(samples, train_set, test_refs=test_sents)
            res["results"].append({"temperature": T, "free": rep})
            print(f"T={T}: test-BLEU-4 {rep['test_bleu4']:.4f} self-BLEU-4 {rep['self_bleu4']:.4f} "
                  f"D-2 {rep['distinct2']:.3f}", flush=True)
    tmp = out_path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
