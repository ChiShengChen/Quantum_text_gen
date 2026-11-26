"""Statistics of the trained QKSAN quantum kernel as the number of qubits grows.

    python -m qnlg.kernel_diag --out asoc_paper/revision_src

For each checkpoint of the qubit-scaling study (Short Stories), the kernel entries
Gamma_ij = |<phi(q_i)|phi(k_j)>|^2 for all causal pairs j < i of one test batch (128 sentences)
are recorded in every layer and head; the table reports their mean and standard deviation
next to 1/2^n, the mean fidelity of two Haar-random n-qubit states.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from . import models as Mo
from .data import Vocab, load_corpus, make_loader
from .models import Config, build_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def kernel_entries(model, x):
    caught = []
    orig = Mo.QuantumKernelAttention.forward

    def hook(self, xx, mask):
        B, T, _ = xx.shape
        q, k, _v = (t.reshape(B, T, self.h, self.dh) for t in self.qkv(xx).chunk(3, dim=-1))
        a = np.pi * torch.tanh(torch.einsum("bthd,hdn->hbtn", torch.cat([q, k], 1), self.feat_proj))
        if self.classical:
            u = torch.tanh(torch.einsum("hbtn,hnm->hbtm", a, self.feat_W) + self.feat_b[:, None, None])
            G = torch.exp(-torch.cdist(u[:, :, :T], u[:, :, T:]) ** 2)
        else:
            st = Mo.reupload_state_grouped(a.reshape(self.h, -1, self.n), self.feat_w).reshape(self.h, B, 2 * T, -1)
            ov = torch.einsum("hbik,hbjk->hbij", st[:, :, :T].conj(), st[:, :, T:])
            G = (ov.real ** 2 + ov.imag ** 2).float()
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
        caught.append(G[..., tri].flatten())
        return orig(self, xx, mask)

    Mo.QuantumKernelAttention.forward = hook
    try:
        with torch.no_grad():
            model(x)
    finally:
        Mo.QuantumKernelAttention.forward = orig
    return torch.cat(caught)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "asoc_paper", "revision_src"))
    ap.add_argument("--dataset", default="short_stories")
    a = ap.parse_args()
    splits, _ = load_corpus(a.dataset)
    vocab = Vocab(splits["train"], 2)
    x, _ = next(iter(make_loader(splits["test"], vocab, 64, 128, False)))
    res, lines = {}, []
    for n in (2, 4, 6, 8):
        stats, ppl = [], []
        for f in sorted(glob.glob(os.path.join(ROOT, "experiments_rev", "qubits", a.dataset, "qksan", f"seed*_n{n}.pt"))):
            ck = torch.load(f, map_location="cpu", weights_only=True)
            model = build_model("qksan", Config(**ck["config"]))
            model.load_state_dict(ck["state_dict"])
            model.eval()
            G = kernel_entries(model, x)
            stats.append((G.mean().item(), G.std().item()))
            js = f[:-3] + ".json"
            if os.path.exists(js):
                ppl.append(json.load(open(js))["test"]["ppl"])
        if not stats:
            continue
        s = np.array(stats)
        res[n] = {"mean": float(s[:, 0].mean()), "std": float(s[:, 1].mean()), "haar": 2.0 ** -n,
                  "ppl": float(np.mean(ppl)) if ppl else None, "checkpoints": len(stats)}
        lines.append(f"{n} & {2 ** n} & {s[:, 0].mean():.3f} & {2.0 ** -n:.3f} & {s[:, 1].mean():.3f} & "
                     f"{np.mean(ppl):.2f} \\\\")
        print(n, res[n])
    # quantum vs classical-surrogate kernel at the default size, Haiku (checkpoints of experiments_rev/ckpt)
    hs, _ = load_corpus("haiku")
    hv = Vocab(hs["train"], 2)
    hx, _ = next(iter(make_loader(hs["test"], hv, 64, 128, False)))
    for name in ("qksan", "qksan_c"):
        vals = []
        for f in sorted(glob.glob(os.path.join(ROOT, "experiments_rev", "ckpt", "haiku", name, "seed*.pt"))):
            ck = torch.load(f, map_location="cpu", weights_only=True)
            model = build_model(name, Config(**ck["config"]))
            model.load_state_dict(ck["state_dict"])
            model.eval()
            G = kernel_entries(model, hx)
            vals.append((G.mean().item(), torch.log(G + 1e-6).std().item()))
        if vals:
            v = np.array(vals)
            res[f"haiku_{name}"] = {"mean": float(v[:, 0].mean()), "std_log": float(v[:, 1].mean()), "checkpoints": len(vals)}
            print(name, res[f"haiku_{name}"])
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "tables", "kernel_diag.json"), "w"), indent=1)
    open(os.path.join(a.out, "tables", "kernel.tex"), "w").write(f"""\\begin{{table}}[t]
\\centering
\\caption{{Concentration of the trained QKSAN quantum kernel (Short Stories, qubit-scaling models, mean over seeds). Mean and standard deviation of the kernel entries $\\Gamma_{{ij}}$ over all causal query--key pairs of one test batch, all layers and heads, compared with $1/2^n$, the mean fidelity of two Haar-random $n$-qubit states; last column: test perplexity.}}
\\label{{tab:kernel}}
\\small
\\begin{{tabular}}{{rrcccc}}
\\toprule
$n$ & $\\dim\\mathcal{{H}}$ & mean $\\Gamma_{{ij}}$ & $1/2^n$ & s.d.\\ $\\Gamma_{{ij}}$ & PPL \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    print("wrote tables/kernel.tex")


if __name__ == "__main__":
    main()
