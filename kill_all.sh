# this will kill all GPU compute processes except Xorg
nvidia-smi --query-compute-apps=pid,process_name \
  --format=csv,noheader | \
  awk -F, '$2 !~ /Xorg/ { print $1 }' | \
  xargs --no-run-if-empty kill -9

# also kill tmux session named train_gpu if it exists
tmux kill-session -t train_gpu 2>/dev/null || true