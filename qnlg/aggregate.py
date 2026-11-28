"""Aggregate experiments_rev/* into the LaTeX tables and PDF figures of the revised paper.

    python -m qnlg.aggregate --out asoc_paper/revision_src

Every number in the paper's result tables and every example sentence is read
from the per-run JSON logs written by qnlg.train, so tables and examples cannot
drift apart.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXP = os.path.join(ROOT, "experiments_rev")

DATASETS = ["simple_sentences", "quantum_phrases", "proverbs", "short_stories", "haiku", "ptb"]
DS_LABEL = {"simple_sentences": "Simple", "quantum_phrases": "Quantum", "proverbs": "Proverbs",
            "short_stories": "Stories", "haiku": "Haiku", "ptb": "PTB"}
DS_LONG = {"simple_sentences": "Simple Sentences", "quantum_phrases": "Quantum Phrases", "proverbs": "Proverbs",
           "short_stories": "Short Stories", "haiku": "Haiku", "ptb": "Penn Treebank"}
# one fixed order used in every table and figure (reviewer comment 2.5a)
MODELS = ["mlp", "transformer", "rwkv", "qasa", "qasa_c", "qksan", "qksan_c", "qrwkv", "qrwkv_c"]
LABEL = {"mlp": "MLP-LM", "transformer": "Transformer", "rwkv": "RWKV", "qasa": "QASA", "qasa_c": "QASA-CS",
         "qksan": "QKSAN", "qksan_c": "QKSAN-CS", "qrwkv": "QRWKV", "qrwkv_c": "QRWKV-CS"}
PAIRS = [("qasa", "qasa_c", "transformer"), ("qksan", "qksan_c", "transformer"), ("qrwkv", "qrwkv_c", "rwkv")]

C_BASE, C_Q, C_CS = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(group):
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(EXP, group, "*", "*", "*.json")):
        d = json.load(open(f))
        runs[(d["args"]["dataset"], d["args"]["model"], d["args"].get("tag", ""))].append(d)
    return runs


def get(d, path):
    for p in path.split("."):
        d = d[p]
    return d


def ms(runs, path):
    v = np.array([get(r, path) for r in runs], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), v


def fmt(m, s=None, nd=2, bold=False):
    t = f"{m:.{nd}f}" if s is None else f"{m:.{nd}f}$\\pm${s:.{nd}f}"
    return f"\\textbf{{{t}}}" if bold else t


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    print("wrote", os.path.relpath(path, ROOT))


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def table_datasets(out):
    rows = []
    for ds in DATASETS:
        m = json.load(open(os.path.join(ROOT, "corpora", ds, "meta.json")))
        tr = [l.split() for l in open(os.path.join(ROOT, "corpora", ds, "train.txt"))]
        from collections import Counter
        c = Counter(w for s in tr for w in s)
        vocab = len(c) if ds == "ptb" else sum(k >= 2 for k in c.values())
        src = {"simple_sentences": "synthetic (ours)", "quantum_phrases": "synthetic (ours)",
               "proverbs": "Wiktionary", "short_stories": "TinyStories", "haiku": "statworx/haiku",
               "ptb": "PTB (Mikolov)"}[ds]
        s = m["sizes"]
        rows.append(f"{DS_LONG[ds]} & {src} & {s['train']:,} & {s['valid']:,} & {s['test']:,} & "
                    f"{m['avg_len']:.1f} & {m['max_len']} & {vocab + 4:,} & {m['test_in_train']} \\\\")
    body = "\n".join(rows)
    write(os.path.join(out, "tables", "datasets.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Corpora used in the revised experiments. Sentences are de-duplicated before splitting (except PTB, which uses its standard splits); ``Len'' is the mean number of word tokens per training sentence; ``Vocab'' counts training word types with frequency $\\ge 2$ (all types for PTB) plus four special tokens; ``Overlap'' is the number of test sentences that also occur in the training split.}}
\\label{{tab:datasets}}
\\small
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
Corpus & Source & Train & Valid & Test & Len & Max & Vocab & Overlap \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_metric(runs, out, path, name, caption, label, nd=2, lower=True):
    head = " & ".join(DS_LABEL[d] for d in DATASETS)
    best = {}
    for ds in DATASETS:
        vals = {m: ms(runs[(ds, m, "")], path)[0] for m in MODELS if runs.get((ds, m, ""))}
        if vals:
            best[ds] = (min if lower else max)(vals, key=vals.get)
    lines = []
    for m in MODELS:
        cells = []
        for ds in DATASETS:
            r = runs.get((ds, m, ""))
            if not r:
                cells.append("--")
                continue
            mu, sd, _ = ms(r, path)
            cells.append(fmt(mu, sd, nd, bold=best.get(ds) == m))
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
        if m in ("rwkv", "qasa_c", "qksan_c"):
            lines.append("\\midrule")
    n_seeds = max(len(v) for k, v in runs.items() if k[2] == "")
    write(os.path.join(out, "tables", f"{name}.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption} Mean $\\pm$ standard deviation over {n_seeds} random seeds; best mean per corpus in bold. ``-CS'' denotes the classical-surrogate ablation of the hybrid model directly above it.}}
\\label{{{label}}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{'c' * len(DATASETS)}}}
\\toprule
Model & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_per_dataset(runs, out):
    cols = [("test.ppl", "PPL", 2), ("test.acc", "Acc", 3), ("test.brier", "Brier", 3),
            ("prompted.bleu1", "B-1", 3), ("prompted.bleu2", "B-2", 3), ("prompted.bleu4", "B-4", 3),
            ("free.test_bleu4", "T-B4", 3), ("free.self_bleu4", "S-B4", 3), ("free.distinct1", "D-1", 3),
            ("free.distinct2", "D-2", 3), ("free.seq_rep2", "Rep", 3), ("free.novelty", "Nov", 2),
            ("free.mean_len", "Len", 1)]
    blocks = []
    for ds in DATASETS:
        lines = []
        for m in MODELS:
            r = runs.get((ds, m, ""))
            if not r:
                continue
            lines.append(f"{LABEL[m]} & " + " & ".join(fmt(ms(r, p)[0], None, nd) for p, _, nd in cols) + " \\\\")
        if lines:
            blocks.append(f"\\midrule\n\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{DS_LONG[ds]}}}}} \\\\\n"
                          + "\n".join(lines))
    head = " & ".join(c for _, c, _ in cols)
    write(os.path.join(out, "tables", "per_dataset_full.tex"), f"""\\begin{{table}}[p]
\\centering
\\caption{{Full results per corpus (means over seeds). Language modelling on the test split: perplexity (PPL), next-token accuracy (Acc), multi-class Brier score. Prompted continuation (2-word prompt, greedy decoding): corpus BLEU-1/2/4 (B-$n$) against the reference continuation. Free generation (1000 nucleus samples, $p=0.9$): Test-BLEU-4 against the test set (T-B4, quality), Self-BLEU-4 (S-B4, lower = more diverse), Distinct-1/2 (D-$n$), within-sentence repeated-bigram rate (Rep), fraction of samples not in the training set (Nov), and mean length in tokens (Len).}}
\\label{{tab:full}}
\\tiny
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\begin{{tabular}}{{l{'r' * len(cols)}}}
\\toprule
Model & {head} \\\\
{chr(10).join(blocks)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def table_ablation(runs, out):
    """Quantum model vs (a) its classical surrogate and (b) its fully classical parent, test PPL."""
    rows, pv, keys = [], [], []
    for q, cs, parent in PAIRS:
        for ds in DATASETS:
            if not (runs.get((ds, q, "")) and runs.get((ds, cs, "")) and runs.get((ds, parent, ""))):
                continue
            _, _, vq = ms(runs[(ds, q, "")], "test.ppl")
            _, _, vc = ms(runs[(ds, cs, "")], "test.ppl")
            _, _, vp = ms(runs[(ds, parent, "")], "test.ppl")
            for ref, vr in (("cs", vc), ("parent", vp)):
                t = stats.ttest_ind(vq, vr, equal_var=False)
                pv.append(t.pvalue if np.isfinite(t.pvalue) else 1.0)
                keys.append((q, ds, ref, vq.mean() - vr.mean(), 100 * (vq.mean() - vr.mean()) / vr.mean()))
    if not pv:
        return
    adj = holm(np.array(pv))
    table = defaultdict(dict)
    for (q, ds, ref, diff, rel), p in zip(keys, adj):
        star = "$^{*}$" if p < 0.05 else ""
        table[(q, ref)][ds] = f"{diff:+.2f} ({rel:+.1f}\\%){star}"
    lines = []
    for q, cs, parent in PAIRS:
        for ref, name in (("cs", f"{LABEL[q]} $-$ {LABEL[cs]}"), ("parent", f"{LABEL[q]} $-$ {LABEL[parent]}")):
            lines.append(name + " & " + " & ".join(table[(q, ref)].get(ds, "--") for ds in DATASETS) + " \\\\")
    sig = sum(p < 0.05 for p in adj)
    write(os.path.join(out, "tables", "ablation.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Effect of the quantum circuit on test perplexity: difference in mean PPL between each hybrid model and (i) its classical surrogate (same architecture, every circuit replaced by a $\\tanh$-linear map of the same width) and (ii) its fully classical parent architecture. Negative values favour the hybrid model. $^{{*}}$: Welch $t$-test over seeds significant at $\\alpha=0.05$ after Holm--Bonferroni correction across all {len(adj)} comparisons ({sig} significant).}}
\\label{{tab:ablation}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{l{'c' * len(DATASETS)}}}
\\toprule
Comparison & {" & ".join(DS_LABEL[d] for d in DATASETS)} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    json.dump([{"model": q, "dataset": ds, "ref": ref, "diff": diff, "rel_pct": rel, "p_holm": float(p)}
               for (q, ds, ref, diff, rel), p in zip(keys, adj)],
              open(os.path.join(out, "tables", "ablation_stats.json"), "w"), indent=1)


def table_cost(runs, out):
    timing = load("timing")  # dedicated benchmark on an otherwise idle GPU, if available
    lines = []
    for m in MODELS:
        r_ptb = runs.get(("ptb", m, ""))
        r_ss = runs.get(("short_stories", m, ""))
        r_any = r_ptb or r_ss
        if not r_any:
            continue
        d = r_any[0]
        body = d["params"]["body"]
        circ = d["params"]["circuit"]
        hd = d["hilbert_dims"]
        hil = "--" if not hd else (f"{hd[0]}" if len(set(hd)) == 1 else "/".join(map(str, sorted(set(hd)))))
        ncirc = len(hd) if hd else 0
        if m == "qksan":  # feature-map and value circuit for every head of every layer
            ncirc = 2 * d["config"]["n_layers"] * d["config"]["n_heads"]
        qb = d["config"]["n_qubits"] if m not in ("mlp", "transformer", "rwkv") else "--"
        if m == "qasa":
            qb = f"{qb}+1"
        def sec(r, ds):
            t = timing.get((ds, m, "timing"))
            if t:  # epochs 2-3 of the isolated benchmark
                return f"{np.median([h['sec'] for h in t[0]['history'][1:]]):.1f}"
            if not r:
                return "--"
            v = [np.median([h["sec"] for h in x["history"][1:]] or [x["history"][0]["sec"]]) for x in r]
            return f"{np.mean(v):.1f}"
        def ep(r):
            if not r:
                return "--"
            return f"{np.mean([x['best_epoch'] for x in r]):.1f}"
        lines.append(f"{LABEL[m]} & {body:,} & {circ} & {qb} & {hil} & {ncirc} & {sec(r_ss, 'short_stories')} & {sec(r_ptb, 'ptb')} & "
                     f"{ep(r_ss)} & {ep(r_ptb)} \\\\")
        if m in ("rwkv", "qasa_c", "qksan_c"):
            lines.append("\\midrule")
    write(os.path.join(out, "tables", "cost.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Model size and cost. ``Body'' counts trainable parameters excluding the vocabulary-dependent embedding and output layers (which are identical across models); ``Circ.'' is the number of trainable circuit angles; $n$ is the number of qubits per circuit (QASA: 8 data qubits plus one ancilla) and $\\dim\\mathcal{{H}}=2^{{n}}$ the dimension of the simulated Hilbert space; \\#VQC is the number of distinct circuits in the model. Wall-clock seconds per training epoch on one RTX~3090 used by no other job (median of epochs 2--3 of a dedicated 3-epoch benchmark run) for Short Stories (8k sentences) and PTB (42k sentences), and the epoch with the best validation loss (mean over seeds; early stopping with patience 5, at most 100 epochs).}}
\\label{{tab:cost}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lrrcccrrrr}}
\\toprule
 & & & & & & \\multicolumn{{2}}{{c}}{{s / epoch}} & \\multicolumn{{2}}{{c}}{{best epoch}} \\\\
\\cmidrule(lr){{7-8}}\\cmidrule(lr){{9-10}}
Model & Body & Circ. & $n$ & $\\dim\\mathcal{{H}}$ & \\#VQC & Stories & PTB & Stories & PTB \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_tf_sweep(out):
    runs = load("tf_sweep")
    if not runs:
        return
    ds_list = [d for d in ["simple_sentences", "short_stories", "haiku"] if any(k[0] == d for k in runs)]
    Ls, ds_ = [1, 2, 4, 6], [32, 64, 128, 256]
    lines = []
    for d in ds_:
        for L in Ls:
            tag = f"d{d}_L{L}"
            cells, body = [], None
            for ds in ds_list:
                r = runs.get((ds, "transformer", tag))
                if not r:
                    cells += ["--", "--"]
                    continue
                body = r[0]["params"]["body"]
                mu, sd, _ = ms(r, "test.ppl")
                cells += [fmt(mu, sd, 2), fmt(ms(r, "prompted.bleu2")[0], None, 3)]
            lines.append(f"{d} & {L} & {body:,} & " + " & ".join(cells) + " \\\\" if body else
                         f"{d} & {L} & -- & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_LONG[d]}}}" for d in ds_list)
    sub = " & ".join("PPL & B-2" for _ in ds_list)
    cm = "".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(ds_list)))
    write(os.path.join(out, "tables", "tf_sweep.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Effect of the Transformer configuration (width $d$, depth $L$; $h=4$ heads, $d_{{\\rm ff}}=4d$) on test perplexity (mean $\\pm$ s.d., 3 seeds) and prompted BLEU-2. The original submission used $d=256$, $L=6$; the matched baseline of Tables~\\ref{{tab:ppl}}--\\ref{{tab:ablation}} uses $d=64$, $L=2$.}}
\\label{{tab:tfsweep}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{rrr{'cc' * len(ds_list)}}}
\\toprule
 & & & {hdr} \\\\
{cm}
$d$ & $L$ & Body & {sub} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_qubits(out):
    runs = load("qubits")
    if not runs:
        return
    lines = []
    for m in ["qasa", "qksan", "qrwkv"]:
        for n in [2, 4, 6, 8]:
            cells = []
            for ds in ["simple_sentences", "short_stories"]:
                r = runs.get((ds, m, f"n{n}"))
                if not r:
                    cells += ["--", "--"]
                    continue
                mu, sd, _ = ms(r, "test.ppl")
                sec = np.mean([np.median([h["sec"] for h in x["history"][1:]] or [x["history"][0]["sec"]]) for x in r])
                cells += [fmt(mu, sd, 2), f"{sec:.1f}"]
            dim = 2 ** (n + 1) if m == "qasa" else 2 ** n
            lines.append(f"{LABEL[m] if n == 2 else ''} & {n}{'+1' if m == 'qasa' else ''} & {dim} & "
                         + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write(os.path.join(out, "tables", "qubits.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Qubit-count scaling: test perplexity (mean $\\pm$ s.d., 3 seeds) and seconds per training epoch as the number of qubits $n$ per circuit, and hence the simulated Hilbert-space dimension $2^n$, grows. All other settings as in Table~\\ref{{tab:cost}}.}}
\\label{{tab:qubits}}
\\scriptsize
\\begin{{tabular}}{{lrrcccc}}
\\toprule
 & & & \\multicolumn{{2}}{{c}}{{Simple Sentences}} & \\multicolumn{{2}}{{c}}{{Short Stories}} \\\\
\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}
Model & $n$ & $\\dim\\mathcal{{H}}$ & PPL & s/epoch & PPL & s/epoch \\\\
\\midrule
{chr(10).join(lines[:-1])}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")


def table_examples(runs, out, datasets=("short_stories", "haiku", "proverbs"), k=2):
    blocks = []
    for ds in datasets:
        r0 = {m: [x for x in runs.get((ds, m, ""), []) if x["args"]["seed"] == 0] for m in MODELS}
        if not all(r0.values()):
            continue
        ex = {m: r0[m][0]["examples"]["prompted"] for m in MODELS}
        for i in range(k):
            ref = ex["transformer"][i]
            lines = [f"\\multicolumn{{2}}{{l}}{{\\textit{{{DS_LONG[ds]}}} --- prompt: ``{esc(ref['prompt'])}''; "
                     f"reference: ``{esc(ref['reference'])}''}} \\\\"]
            for m in MODELS:
                assert ex[m][i]["prompt"] == ref["prompt"]
                gen = esc(ex[m][i]["generated"]) or "\\textit{(empty)}"
                lines.append(f"\\quad {LABEL[m]} & {gen} \\\\")
            blocks.append("\n".join(lines))
    sep = "\n\\midrule\n"
    body = sep.join(blocks)
    write(os.path.join(out, "tables", "examples.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Greedy continuations of the same two-word test prompts by every model (seed 0), copied verbatim from the evaluation logs that produce Tables~\\ref{{tab:ppl}}, \\ref{{tab:bleu2}} and~\\ref{{tab:full}}. \\texttt{{<unk>}} marks out-of-vocabulary words.}}
\\label{{tab:examples}}
\\scriptsize
\\begin{{tabular}}{{lp{{0.72\\linewidth}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})
    return plt


def fig_convergence(runs, out, datasets=("short_stories", "ptb")):
    plt = _style()
    fig, axes = plt.subplots(len(datasets), 3, figsize=(7.0, 2.0 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, (q, cs, parent) in enumerate(PAIRS):
            ax = axes[i][j]
            for m, col, ls in ((parent, C_BASE, "-"), (q, C_Q, "-"), (cs, C_CS, "--")):
                r = runs.get((ds, m, ""))
                if not r:
                    continue
                L = max(len(x["history"]) for x in r)
                M = np.full((len(r), L), np.nan)
                for a, x in enumerate(r):
                    v = [h["val_nll"] for h in x["history"]]
                    M[a, :len(v)] = v
                mu = np.nanmean(M, 0)
                ax.plot(np.arange(1, L + 1), mu, color=col, ls=ls, label=LABEL[m])
            ax.set_title(f"{DS_LONG[ds]}: {LABEL[q]} family", fontsize=7.5, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("validation NLL (nats/token)")
            if i == len(datasets) - 1:
                ax.set_xlabel("epoch")
            ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    p = os.path.join(out, "figures", "convergence.pdf")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p)
    print("wrote", os.path.relpath(p, ROOT))


def fig_quality_diversity(runs, out):
    """Test-BLEU-4 (quality) vs Self-BLEU-4 (diversity) per corpus: hybrid vs classical families."""
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    marker = {"mlp": "s", "transformer": "o", "rwkv": "D", "qasa": "^", "qksan": "v", "qrwkv": "P",
              "qasa_c": "^", "qksan_c": "v", "qrwkv_c": "P"}
    for ax, ds in zip(axes.flat, DATASETS):
        for m in MODELS:
            r = runs.get((ds, m, ""))
            if not r:
                continue
            x, y = ms(r, "free.self_bleu4")[0], ms(r, "free.test_bleu4")[0]
            col = C_CS if m.endswith("_c") else (C_Q if m.startswith("q") else C_BASE)
            ax.scatter(x, y, s=34, marker=marker[m], color=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(LABEL[m], (x, y), fontsize=5.2, color=INK2, xytext=(3, 2), textcoords="offset points")
        ax.set_title(DS_LONG[ds], fontsize=7.5, loc="left", color=INK)
        ax.set_xlabel("Self-BLEU-4 (lower = more diverse)")
        ax.set_ylabel("Test-BLEU-4 (quality)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
         ((C_BASE, "classical baseline"), (C_Q, "hybrid quantum"), (C_CS, "classical surrogate"))]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(out, "figures", "quality_diversity.pdf")
    fig.savefig(p)
    print("wrote", os.path.relpath(p, ROOT))


# ----------------------------------------------------------------------------
# Additional studies: width, shots, temperature
# ----------------------------------------------------------------------------

WIDTH_DS = ["simple_sentences", "proverbs", "short_stories"]


def width_runs(main):
    """ppl samples per (dataset, model, d); d=64 taken from the main grid (seeds 0-2 for parity)."""
    w = load("width")
    out = {}
    for ds in WIDTH_DS:
        for m in MODELS:
            for d in (16, 32):
                r = w.get((ds, m, f"d{d}"))
                if r:
                    out[(ds, m, d)] = r
            r = [x for x in main.get((ds, m, ""), []) if x["args"]["seed"] in (0, 1, 2)]
            if r:
                out[(ds, m, 64)] = r
    return out


def table_width(main, out):
    W = width_runs(main)
    if not any(k[2] != 64 for k in W):
        return
    rows, pv, keys = [], [], []
    for q, cs, parent in PAIRS:
        for ds in WIDTH_DS:
            for d in (16, 32, 64):
                if not all((ds, m, d) in W for m in (q, cs, parent)):
                    continue
                vq = ms(W[(ds, q, d)], "test.ppl")[2]
                for ref, mref in (("cs", cs), ("parent", parent)):
                    vr = ms(W[(ds, mref, d)], "test.ppl")[2]
                    t = stats.ttest_ind(vq, vr, equal_var=False)
                    pv.append(t.pvalue if np.isfinite(t.pvalue) else 1.0)
                    keys.append((q, ds, d, ref, 100 * (vq.mean() - vr.mean()) / vr.mean()))
    if not pv:
        return
    adj = holm(np.array(pv))
    cell = {}
    for (q, ds, d, ref, rel), p in zip(keys, adj):
        cell[(q, ds, d, ref)] = f"{rel:+.1f}{'$^{*}$' if p < 0.05 else ''}"
    lines = []
    for q, cs, parent in PAIRS:
        for ref, name in (("cs", f"{LABEL[q]} vs {LABEL[cs]}"), ("parent", f"{LABEL[q]} vs {LABEL[parent]}")):
            lines.append(name + " & " + " & ".join(cell.get((q, ds, d, ref), "--") for ds in WIDTH_DS
                                                    for d in (16, 32, 64)) + " \\\\")
    hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS_LONG[ds]}}}" for ds in WIDTH_DS)
    sub = " & ".join("16 & 32 & 64" for _ in WIDTH_DS)
    cm = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(WIDTH_DS)))
    sig = int((adj < 0.05).sum())
    write(os.path.join(out, "tables", "width.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Narrow models: relative difference in test perplexity (\\%) between each hybrid model and its classical surrogate or classical parent at width $d=16, 32, 64$ (3 seeds each; negative values favour the hybrid model). $^{{*}}$: significant after Holm--Bonferroni correction over all {len(adj)} comparisons ({sig} significant).}}
\\label{{tab:width}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{l{'ccc' * len(WIDTH_DS)}}}
\\toprule
 & {hdr} \\\\
{cm}
Comparison, $d=$ & {sub} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    json.dump([{"model": q, "dataset": ds, "d": d, "ref": ref, "rel_pct": rel, "p_holm": float(p)}
               for (q, ds, d, ref, rel), p in zip(keys, adj)],
              open(os.path.join(out, "tables", "width_stats.json"), "w"), indent=1)


def fig_width(main, out):
    W = width_runs(main)
    if not any(k[2] != 64 for k in W):
        return
    plt = _style()
    fig, axes = plt.subplots(len(WIDTH_DS), 3, figsize=(7.0, 1.9 * len(WIDTH_DS)), squeeze=False)
    for i, ds in enumerate(WIDTH_DS):
        for j, (q, cs, parent) in enumerate(PAIRS):
            ax = axes[i][j]
            for m, col, ls, mk in ((parent, C_BASE, "-", "o"), (q, C_Q, "-", "^"), (cs, C_CS, "--", "s")):
                xs, ys, es = [], [], []
                for d in (16, 32, 64):
                    if (ds, m, d) in W:
                        mu, sd, _ = ms(W[(ds, m, d)], "test.ppl")
                        xs.append(d); ys.append(mu); es.append(sd)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker=mk, ms=4, capsize=2, label=LABEL[m])
            ax.set_xscale("log", base=2)
            ax.set_xticks([16, 32, 64]); ax.set_xticklabels(["16", "32", "64"])
            ax.set_title(f"{DS_LONG[ds]}: {LABEL[q]} family", fontsize=7.5, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("test perplexity")
            if i == len(WIDTH_DS) - 1:
                ax.set_xlabel("model width $d$")
            ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    p = os.path.join(out, "figures", "width.pdf")
    fig.savefig(p)
    print("wrote", os.path.relpath(p, ROOT))


def load_eval(group):
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(EXP, group, "*", "*", "seed*.json")):
        d = json.load(open(f))
        runs[(d["args"]["dataset"], d["args"]["model"])].append(d)
    return runs


def shots_outputs(out):
    R = load_eval("shots")
    if not R:
        return
    ds_list = [d for d in ("haiku", "ptb") if any(k[0] == d for k in R)]
    models = ["qasa", "qksan", "qrwkv"]
    shots = [None, 4096, 1024, 256, 64]
    lines = []
    for ds in ds_list:
        for m in models:
            r = R.get((ds, m))
            if not r:
                continue
            cells = []
            for S in shots:
                ppl = [next(x for x in run["results"] if x["shots"] == S)["test"]["ppl"] for run in r]
                b2 = [next(x for x in run["results"] if x["shots"] == S)["prompted"]["bleu2"] for run in r]
                cells.append(f"{np.mean(ppl):.2f} / {np.mean(b2):.3f}")
            lines.append(f"{DS_LONG[ds]} & {LABEL[m]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write(os.path.join(out, "tables", "shots.tex"), f"""\\begin{{table}}[t]
\\centering
\\caption{{Finite-shot evaluation of the trained hybrid models: test perplexity / prompted BLEU-2 when every expectation value (and, for QKSAN, every kernel entry) is estimated from $S$ measurement shots instead of computed exactly (means over 5 seeds). Pauli-$Z$ expectations are estimated from $S$ jointly sampled bitstrings; kernel entries from the fraction of $S$ compute--uncompute runs returning $\\ket{{0}}^{{\\otimes n}}$. No retraining.}}
\\label{{tab:shots}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{llccccc}}
\\toprule
Corpus & Model & exact & $S=4096$ & $S=1024$ & $S=256$ & $S=64$ \\\\
\\midrule
{chr(10).join(lines[:-1])}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def fig_temps(out):
    R = load_eval("temps")
    if not R:
        return
    plt = _style()
    ds_list = [d for d in ("haiku", "ptb") if any(k[0] == d for k in R)]
    fig, axes = plt.subplots(len(ds_list), 3, figsize=(7.0, 2.2 * len(ds_list)), squeeze=False)
    for i, ds in enumerate(ds_list):
        for j, (q, cs, parent) in enumerate(PAIRS):
            ax = axes[i][j]
            for m, col, ls, mk in ((parent, C_BASE, "-", "o"), (q, C_Q, "-", "^"), (cs, C_CS, "--", "s")):
                r = R.get((ds, m))
                if not r:
                    continue
                temps = [x["temperature"] for x in r[0]["results"]]
                xs = [np.mean([run["results"][k]["free"]["self_bleu4"] for run in r]) for k in range(len(temps))]
                ys = [np.mean([run["results"][k]["free"]["test_bleu4"] for run in r]) for k in range(len(temps))]
                ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.5, label=LABEL[m])
                if m == parent:
                    for t, x, y in zip(temps, xs, ys):
                        if t in (min(temps), max(temps)):
                            ax.annotate(f"T={t:g}", (x, y), fontsize=5.5, color=INK2, xytext=(3, -8),
                                        textcoords="offset points")
            ax.set_title(f"{DS_LONG[ds]}: {LABEL[q]} family", fontsize=7.5, loc="left", color=INK)
            ax.set_xlabel("Self-BLEU-4 (lower = more diverse)")
            if j == 0:
                ax.set_ylabel("Test-BLEU-4 (quality)")
            ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    p = os.path.join(out, "figures", "temperature.pdf")
    fig.savefig(p)
    print("wrote", os.path.relpath(p, ROOT))


def repro_check(main, out):
    """Re-trained Haiku runs (experiments_rev/ckpt) vs the main grid, same seeds: run-to-run variation."""
    ck = load("ckpt")
    rows = {}
    for m in MODELS:
        a = {r["args"]["seed"]: r["test"]["ppl"] for r in main.get(("haiku", m, ""), [])}
        b = {r["args"]["seed"]: r["test"]["ppl"] for r in ck.get(("haiku", m, ""), [])}
        common = sorted(set(a) & set(b))
        if common:
            rel = [100 * abs(a[s] - b[s]) / a[s] for s in common]
            rows[m] = {"seeds": common, "max_abs_rel_diff_pct": max(rel), "mean_abs_rel_diff_pct": float(np.mean(rel))}
    if rows:
        json.dump(rows, open(os.path.join(out, "tables", "repro_haiku.json"), "w"), indent=1)
        print("repro:", {m: round(v["max_abs_rel_diff_pct"], 3) for m, v in rows.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "asoc_paper", "revision_src"))
    a = ap.parse_args()
    runs = load("main")
    table_datasets(a.out)
    table_metric(runs, a.out, "test.ppl", "ppl", "Test perplexity (lower is better).", "tab:ppl")
    table_metric(runs, a.out, "prompted.bleu2", "bleu2",
                 "Prompted continuation: corpus BLEU-2 of greedy continuations of two-word test prompts against "
                 "the reference continuations (higher is better).", "tab:bleu2", nd=3, lower=False)
    table_per_dataset(runs, a.out)
    table_ablation(runs, a.out)
    table_cost(runs, a.out)
    table_tf_sweep(a.out)
    table_qubits(a.out)
    table_examples(runs, a.out)
    fig_convergence(runs, a.out)
    fig_quality_diversity(runs, a.out)
    table_width(runs, a.out)
    fig_width(runs, a.out)
    shots_outputs(a.out)
    fig_temps(a.out)
    repro_check(runs, a.out)
    # machine-readable summary for the response letter
    summ = {f"{ds}/{m}": {p: ms(runs[(ds, m, "")], p)[:2] for p in
                         ["test.ppl", "test.acc", "test.brier", "prompted.bleu1", "prompted.bleu2", "prompted.bleu4",
                          "free.test_bleu4", "free.self_bleu4", "free.distinct1", "free.distinct2", "free.seq_rep2",
                          "free.consec_rep", "free.novelty", "free.mean_len", "free.single_token_rate",
                          "free.empty_rate", "best_epoch", "epochs_run"]}
            for ds in DATASETS for m in MODELS if runs.get((ds, m, ""))}
    json.dump({k: {p: [float(x) for x in v] for p, v in d.items()} for k, d in summ.items()},
              open(os.path.join(a.out, "tables", "summary.json"), "w"), indent=1)
    print({k: len(v) for k, v in sorted(runs.items())} if os.environ.get("VERBOSE") else f"{len(runs)} cells")


if __name__ == "__main__":
    main()
