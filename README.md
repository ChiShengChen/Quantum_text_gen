# Hybrid quantum–classical language models for text generation

Code, data and logs for *"Hybrid Quantum–Classical Architectures for Natural Language
Generation: A Controlled Multi-Model Benchmark"* (C.-S. Chen and E.-J. Kuo).

The paper compares three hybrid models that contain variational quantum circuits (VQCs),
**QASA**, **QKSAN** and **QRWKV**, against classical language models (**MLP-LM**,
**Transformer**, **RWKV**) and against *classical surrogates* of each hybrid model, in which
every circuit is replaced by a classical map of the same width. All circuits are simulated
exactly (state vectors) in PyTorch on a GPU.

Everything for the paper lives in `qnlg/`. The other scripts in the repository root belong to the
original submission and have been **superseded**; see [Legacy code](#legacy-code) below.

## Layout

| path | content |
|---|---|
| `qnlg/quantum.py` | batched state-vector simulator and the three circuit families (QASA, QRWKV, data re-uploading for QKSAN) |
| `qnlg/models.py` | the nine causal language models (`mlp`, `transformer`, `rwkv`, `qasa`, `qasa_c`, `qksan`, `qksan_c`, `qrwkv`, `qrwkv_c`; `_c` = classical surrogate) |
| `qnlg/prepare_data.py` | builds the six corpora with de-duplicated, disjoint train/valid/test splits |
| `qnlg/data.py` | vocabulary and batching |
| `qnlg/metrics.py` | BLEU (paired and multi-reference), Self-BLEU, Distinct-n, repetition, novelty |
| `qnlg/train.py` | trains one model on one corpus with one seed and writes a JSON log, including generated text |
| `qnlg/aggregate.py` | builds every results table (LaTeX) and figure (PDF) of the paper from the JSON logs |
| `tests/` | simulator checked against PennyLane; metrics checked against NLTK |
| `scripts/submit_all.sh` | every experiment in the paper (main grid, Transformer sweep, qubit scaling) |
| `corpora/` | the processed corpora (Haiku and PTB are rebuilt locally, see below) |
| `experiments_rev/` | per-run JSON logs of all experiments reported in the paper |

## Setup

```bash
conda create -n qnlg python=3.11 -y && conda activate qnlg
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/test_quantum.py && python tests/test_metrics.py
```

If PyTorch fails to import with an `nvJitLink` symbol error, a system CUDA directory on
`LD_LIBRARY_PATH` is shadowing the wheel's libraries: run with `env -u LD_LIBRARY_PATH python ...`.

## Data

```bash
python -m qnlg.prepare_data     # writes corpora/<name>/{train,valid,test}.txt and meta.json
```

| corpus | source | licence | in repo |
|---|---|---|---|
| simple_sentences | synthetic grammar (this work) | released with the paper | yes |
| quantum_phrases | synthetic grammar + 10 hand-written phrases (this work) | released with the paper | yes |
| short_stories | first sentences of TinyStories (validation split) | CDLA-Sharing-1.0 | yes |
| proverbs | Wiktionary *Category:English proverbs* (page titles) | CC BY-SA 4.0 | yes |
| haiku | `statworx/haiku` on Hugging Face, without the Twitter subset | not specified by the source | rebuilt by the script |
| ptb | Penn Treebank, Mikolov et al. pre-processing, standard splits | LDC | rebuilt by the script |

The splits are deterministic (fixed seed), so rebuilding gives identical files. No test sentence
occurs in the training split, except 64 short sentences that are repeated within PTB's standard
splits.

## Reproducing the paper

```bash
# one run
python -m qnlg.train --model qrwkv --dataset haiku --seed 0 --out experiments_rev/main

# everything (submits to the `gq` GPU queue used on our machine; replace `gq submit ... --` by your launcher)
bash scripts/submit_all.sh

# tables and figures -> <out>/tables/*.tex, <out>/figures/*.pdf
python -m qnlg.aggregate --out paper_build
```

Default settings (all models): width 64, 2 blocks, 4 heads, feed-forward width 256, dropout 0.1,
AdamW (lr 1e-3, weight decay 0.01), batch 64, gradient clipping 1.0, up to 100 epochs with early
stopping (patience 5) on validation cross-entropy, seeds 0–4. Circuits: QASA 8 data qubits + 1
ancilla, 4 layers; QKSAN 4 qubits and 2 re-uploading blocks per circuit; QRWKV 4 qubits, 2 layers.

Each `experiments_rev/<group>/<dataset>/<model>/seed<k>.json` log contains the configuration,
parameter counts, Hilbert-space dimensions, the per-epoch learning curve, test metrics, and the
first 30 prompted continuations and free samples verbatim.

## Legacy code

The scripts in the repository root (`comprehensive_evaluation.py`, `qksan_*.py`, `qasa_*.py`,
`qrwkv_model.py`, `transformer_*.py`, `run_*.sh`, …) and `experiments/comprehensive/` produced the
results of the original submission. They contain errors that invalidate those results, which were
found during revision:

* the evaluation compared outputs to five hard-coded reference strings per dataset that were unrelated to the prompts;
* generation echoed the prompt word and usually stopped after it, so most outputs were one token
  long, which made Distinct-1 = 1 and the repetition rate 0 by construction and BLEU-2 = 0;
* perplexity included padding positions;
* the models evaluated as "QASA", "QKSAN" and "MLP" used no context beyond the current token;
* the training data were expanded by duplication, so test sentences also occurred in training.

They are kept only so that the original results can be traced; use `qnlg/` for any new work.
The previous README is in `LEGACY_README_zh.md`.
