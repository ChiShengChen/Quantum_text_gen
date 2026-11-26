#!/bin/bash
# Additional studies of the revision (run after scripts/submit_all.sh):
#   width  - narrow models (d = 16, 32): does the circuit help when classical capacity is scarce?
#   shots  - finite-shot evaluation of trained hybrid models (needs main-grid checkpoints)
#   temps  - quality-diversity across sampling temperatures (needs main-grid checkpoints)
# usage: scripts/submit_extra.sh width | ckpt | eval [corpora, default "haiku ptb"] | timing
set -e
cd "$(dirname "$0")/.."
PY="env -u LD_LIBRARY_PATH ${QNLG_PYTHON:-python}"
case "$1" in
width)
  for ds in simple_sentences proverbs short_stories; do
    for m in mlp transformer rwkv qasa qasa_c qksan qksan_c qrwkv qrwkv_c; do
      cmd=""
      for d in 16 32; do
        cmd+="for s in 0 1 2; do $PY -m qnlg.train --model $m --dataset $ds --seed \$s --d_model $d --tag d$d --out experiments_rev/width --max_epochs 100 --patience 5; done; "
      done
      gq submit -g 2G -c 2 -m 6 -n "qnlg-width-$ds-$m" -- bash -c "$cmd"
    done
  done ;;
ckpt)
  # Haiku models re-trained with saved weights (the main-grid Haiku runs predate checkpoint saving);
  # their JSON logs also serve as a run-to-run reproducibility check against experiments_rev/main.
  for m in mlp transformer rwkv qasa qasa_c qksan qksan_c qrwkv qrwkv_c; do
    seeds="0 1 2"; case $m in qasa|qksan|qrwkv) seeds="0 1 2 3 4";; esac
    gq submit -p 3 -g 2G -c 2 -m 6 -n "qnlg-ckpt-haiku-$m" -- bash -c \
      "for s in $seeds; do $PY -m qnlg.train --model $m --dataset haiku --seed \$s --out experiments_rev/ckpt --max_epochs 100 --patience 5; done"
  done ;;
eval)
  # haiku: checkpoints from experiments_rev/ckpt; ptb: from the main grid (seeds without weights are skipped)
  for ds in ${2:-haiku ptb}; do
    g=2G; root=experiments_rev/ckpt; [ "$ds" = ptb ] && { g=6G; root=experiments_rev/main; }
    for m in qasa qksan qrwkv; do
      gq submit -g $g -c 2 -m 6 -n "qnlg-shots-$ds-$m" -- bash -c \
        "for s in 0 1 2 3 4; do $PY -m qnlg.evaluate shots --root $root --dataset $ds --model $m --seed \$s --shots 64 256 1024 4096; done"
    done
    for m in mlp transformer rwkv qasa qasa_c qksan qksan_c qrwkv qrwkv_c; do
      # first three seeds that have saved weights
      gq submit -g $g -c 2 -m 6 -n "qnlg-temps-$ds-$m" -- bash -c \
        "n=0; for s in 0 1 2 3 4; do [ \$n -ge 3 ] && break; [ -f $root/$ds/$m/seed\$s.pt ] || continue; $PY -m qnlg.evaluate temps --root $root --dataset $ds --model $m --seed \$s --temps 0.6 0.8 1.0 1.2 1.4; n=\$((n+1)); done"
    done
  done ;;
timing)
  # clean wall-clock benchmark: whole GPU (-x), 3 epochs per model, nothing else running
  cmd=""
  for ds in short_stories ptb; do for m in mlp transformer rwkv qasa qasa_c qksan qksan_c qrwkv qrwkv_c; do
    cmd+="echo \"== $ds $m\"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; $PY -m qnlg.train --model $m --dataset $ds --seed 0 --tag timing --out experiments_rev/timing --max_epochs 3 --patience 99 --n_samples 50 --n_prompts 50; "
  done; done
  # -x only waits for a free GPU at start; run this when no other job of ours is queued.
  # The nvidia-smi lines in the log show every process on the GPU before each run.
  gq run -x -c 4 -m 8 -n "qnlg-timing" -t 45m -- bash -c "$cmd" ;;
*) echo "usage: $0 width|ckpt|eval|timing"; exit 1 ;;
esac
