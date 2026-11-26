"""Train one model on one corpus with one seed, then evaluate it.

    python -m qnlg.train --model qrwkv --dataset haiku --seed 0 --out experiments_rev/main

Evaluation (test split):
  * token-level NLL / perplexity (padding excluded, <eos> included), top-1 accuracy, Brier score
  * prompted continuation: first 2 words of each test sentence as prompt, greedy decoding,
    corpus BLEU-1..4 of the continuation against the reference continuation
  * free generation: 1000 nucleus samples (p=0.9) from <bos>: Test-BLEU vs the test set,
    Self-BLEU-4, Distinct-1/2, repetition, novelty w.r.t. the training set, length statistics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .data import load_corpus, make_loader, map_unk, Vocab
from .models import Config, build_model, count_params, QUANTUM_DEFAULTS


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lm_loss(logits, tgt, pad):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=pad)


@torch.no_grad()
def evaluate_tokens(model, loader, pad, device):
    model.eval()
    nll = n = correct = brier = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).float()
        mask = y != pad
        lp = torch.log_softmax(logits, -1)
        tok_lp = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        nll += -(tok_lp[mask]).sum().item()
        correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        p = lp.exp()
        # multi-class Brier score: sum_v (p_v - 1[v=y])^2 = sum p^2 - 2 p_y + 1
        b = (p ** 2).sum(-1) - 2 * tok_lp.exp() + 1
        brier += b[mask].sum().item()
        n += mask.sum().item()
    return {"nll": nll / n, "ppl": math.exp(nll / n), "acc": correct / n, "brier": brier / n, "tokens": int(n)}


@torch.no_grad()
def generate(model, prefix, vocab, max_new, max_len, greedy, top_p=0.9, temperature=1.0, chunk=250, gen=None):
    model.eval()
    device = next(model.parameters()).device
    out = []
    for s in range(0, prefix.size(0), chunk):
        seq = prefix[s:s + chunk].to(device)
        P = seq.size(1)
        done = torch.zeros(seq.size(0), dtype=torch.bool, device=device)
        for _ in range(max_new):
            if seq.size(1) >= max_len:
                break
            logits = model(seq)[:, -1].float() / temperature
            logits[:, vocab.pad] = -float("inf")
            logits[:, vocab.bos] = -float("inf")
            if greedy:
                nxt = logits.argmax(-1)
            else:
                probs = torch.softmax(logits, -1)
                sp, si = probs.sort(-1, descending=True)
                keep = sp.cumsum(-1) - sp < top_p
                sp = sp * keep
                choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen).squeeze(-1)
                nxt = si.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, vocab.pad), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= nxt == vocab.eos
            if done.all():
                break
        out += [vocab.decode(r) for r in seq[:, P:].tolist()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments_rev/main")
    ap.add_argument("--tag", default="")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--n_qubits", type=int, default=None)
    ap.add_argument("--q_depth", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--prompt_words", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    name = f"seed{a.seed}{('_' + a.tag) if a.tag else ''}"
    out_dir = os.path.join(a.out, a.dataset, a.model)
    out_path = os.path.join(out_dir, name + ".json")
    if os.path.exists(out_path):
        print(f"exists, skipping: {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    set_seed(a.seed)
    device = torch.device(a.device)

    splits, meta = load_corpus(a.dataset)
    vocab = Vocab(splits["train"], min_freq=1 if a.dataset == "ptb" else 2)
    tr = make_loader(splits["train"], vocab, a.max_len, a.batch_size, True, a.seed)
    va = make_loader(splits["valid"], vocab, a.max_len, 256, False)
    te = make_loader(splits["test"], vocab, a.max_len, 256, False)

    base = a.model[:-2] if a.model.endswith("_c") else a.model
    qd = QUANTUM_DEFAULTS.get(base, {})
    cfg = Config(vocab_size=len(vocab), d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
                 d_ff=a.d_ff or 4 * a.d_model, dropout=a.dropout, max_len=a.max_len, pad_id=vocab.pad,
                 n_qubits=a.n_qubits or qd.get("n_qubits", 4), q_depth=a.q_depth or qd.get("q_depth", 2))
    model = build_model(a.model, cfg).to(device)
    params = count_params(model)
    hilbert = [m.hilbert_dim for m in model.quantum_modules()]
    if base == "qksan" and not cfg.classical_surrogate:  # feature-map and value circuits per layer
        hilbert += [2 ** cfg.n_qubits] * (2 * cfg.n_layers)
    print(f"{a.model} on {a.dataset} seed {a.seed}: vocab={len(vocab)} params={params} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    hist, best, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    t_train = time.time()
    for ep in range(1, a.max_epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = lm_loss(model(x), y, vocab.pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        v = evaluate_tokens(model, va, vocab.pad, device)
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_nll": v["nll"], "val_ppl": v["ppl"],
                     "sec": time.time() - t0})
        print(f"Epoch {ep}/{a.max_epochs} train {tot / nb:.4f} val_nll {v['nll']:.4f} val_ppl {v['ppl']:.2f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if v["nll"] < best - 1e-4:
            best, best_epoch, bad = v["nll"], ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_sec = time.time() - t_train
    model.load_state_dict(best_state)

    res = {"args": vars(a), "config": cfg.to_dict(), "params": params, "hilbert_dims": hilbert,
           "vocab_size": len(vocab), "corpus_meta": meta, "history": hist, "best_epoch": best_epoch,
           "epochs_run": len(hist), "train_seconds": train_sec}
    res["valid"] = evaluate_tokens(model, va, vocab.pad, device)
    res["test"] = evaluate_tokens(model, te, vocab.pad, device)
    print(f"test ppl {res['test']['ppl']:.3f} acc {res['test']['acc']:.3f}", flush=True)

    # ---- prompted continuation (greedy) ----
    P = a.prompt_words
    test_sents = [map_unk(s, vocab) for s in splits["test"]]
    cand = [s for s in test_sents if len(s) > P][: a.n_prompts]
    prefix = torch.tensor([[vocab.bos] + [vocab.stoi[w] for w in s[:P]] for s in cand])
    t0 = time.time()
    conts = generate(model, prefix, vocab, max_new=a.max_len - P - 1, max_len=a.max_len, greedy=True)
    gen_sec = time.time() - t0
    refs = [s[P:] for s in cand]
    res["prompted"] = M.corpus_bleu_paired(conts, refs, 4)
    res["prompted"].update({"n": len(cand), "exact_match": float(np.mean([c == r for c, r in zip(conts, refs)])),
                            "mean_len": float(np.mean([len(c) for c in conts])),
                            "ref_mean_len": float(np.mean([len(r) for r in refs])), "seconds": gen_sec})
    res["prompted"].update({f"gen_{k}": v for k, v in M.generation_report(
        conts, train_set=set(), test_refs=None).items() if k in ("distinct1", "distinct2", "consec_rep",
                                                                  "seq_rep2", "empty_rate", "single_token_rate")})

    # ---- free generation (nucleus) ----
    g = torch.Generator(device=device)
    g.manual_seed(a.seed + 12345)
    prefix = torch.full((a.n_samples, 1), vocab.bos, dtype=torch.long)
    samples = generate(model, prefix, vocab, max_new=a.max_len - 1, max_len=a.max_len, greedy=False, gen=g)
    train_set = {" ".join(map_unk(s, vocab)) for s in splits["train"]}
    res["free"] = M.generation_report(samples, train_set, test_refs=test_sents)
    print(f"prompted BLEU-2 {res['prompted']['bleu2']:.4f} | free test-BLEU-2 {res['free']['test_bleu2']:.4f} "
          f"self-BLEU-4 {res['free']['self_bleu4']:.4f} distinct-1 {res['free']['distinct1']:.3f}", flush=True)

    res["examples"] = {
        "prompted": [{"prompt": " ".join(s[:P]), "reference": " ".join(r), "generated": " ".join(c)}
                     for s, r, c in list(zip(cand, refs, conts))[:30]],
        "free": [" ".join(s) for s in samples[:30]],
    }
    if device.type == "cuda":
        res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    # weights for later evaluation-only studies (shot noise, decoding sweeps); not tracked in git
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(), "model": a.model,
                "dataset": a.dataset, "seed": a.seed}, os.path.join(out_dir, name + ".pt"))
    tmp = out_path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
