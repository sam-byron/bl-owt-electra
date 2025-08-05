#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-model_babylm_bert.json}"
SESSION="train_gpu"
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)

# start a new detached tmux session
tmux new-session -d -s "$SESSION" -n "gpu${GPUS%%$'\n'*}"

# create one window per GPU
first=true
for GPU in $GPUS; do
  WIN="gpu$GPU"
  if $first; then
    # rename the initial window
    tmux rename-window -t "$SESSION:0" "$WIN"
    first=false
  else
    tmux new-window -t "$SESSION:" -n "$WIN"
  fi

  # send the training loop into that window
  tmux send-keys -t "$SESSION:$WIN" "export CUDA_VISIBLE_DEVICES=$GPU" C-m
  tmux send-keys -t "$SESSION:$WIN" "echo '=== GPU $GPU starting… ==='" C-m
  tmux send-keys -t "$SESSION:$WIN" "\
while true; do \
  python3 transformer_trainer.py --config_path '$CONFIG'; \
  RET=\$?; \
  echo \"[GPU $GPU] exited \$RET\"; \
  [ \$RET -eq 100 ] && break; \
  sleep 30; \
done" C-m
done

# attach to the session
tmux attach -t "$SESSION"