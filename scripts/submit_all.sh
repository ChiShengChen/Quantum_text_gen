#!/bin/bash
# Submit every experiment of the revised paper to the gq GPU queue (one job per dataset x model, seeds inside).
# Results land in experiments_rev/{main,tf_sweep,qubits}/<dataset>/<model>/seed<k>[_tag].json; finished runs are skipped.
set -e
cd "$(dirname "$0")/.."
PY="env -u LD_LIBRARY_PATH ${QNLG_PYTHON:-python}"
COMMON="--max_epochs 100 --patience 5"

# ---- 1. main comparison: 6 corpora x 9 models x 5 seeds ----
for ds in simple_sentences quantum_phrases proverbs short_stories haiku ptb; do
  for m in mlp transformer rwkv qrwkv qrwkv_c qasa qasa_c qksan qksan_c; do
    g=2G; [ "$ds" = ptb ] && g=6G
    cmd="for s in 0 1 2 3 4; do $PY -m qnlg.train --model $m --dataset $ds --seed \$s --out experiments_rev/main $COMMON; done"
    gq submit -g $g -c 2 -m 6 -n "qnlg-main-$ds-$m" -- bash -c "$cmd"
  done
done

# ---- 2. Transformer configuration sweep: d x L, 3 seeds ----
for ds in simple_sentences short_stories haiku; do
  for d in 32 64 128 256; do
    cmd=""
    for L in 1 2 4 6; do
      cmd+="for s in 0 1 2; do $PY -m qnlg.train --model transformer --dataset $ds --seed \$s --d_model $d --n_layers $L --tag d${d}_L${L} --out experiments_rev/tf_sweep $COMMON; done; "
    done
    gq submit -g 2G -c 2 -m 6 -n "qnlg-tf-$ds-d$d" -- bash -c "$cmd"
  done
done

# ---- 3. qubit scaling: n in {2,4,6,8}, 3 seeds ----
for ds in simple_sentences short_stories; do
  for m in qrwkv qasa qksan; do
    cmd=""
    for n in 2 4 6 8; do
      cmd+="for s in 0 1 2; do $PY -m qnlg.train --model $m --dataset $ds --seed \$s --n_qubits $n --tag n$n --out experiments_rev/qubits $COMMON; done; "
    done
    gq submit -g 2G -c 2 -m 6 -n "qnlg-qubits-$ds-$m" -- bash -c "$cmd"
  done
done
