#!/usr/bin/env bash
# Run every line of a queue file ("protocol frontend model aug") sequentially, skipping
# conditions whose result CSV already exists. Usage: bash scripts/run_queue.sh experiments/configs/queue_main.txt
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
while read -r proto fe model aug; do
  [[ -z "$proto" || "$proto" == \#* ]] && continue
  tag="${proto}__${fe}__${model}__${aug}"
  if [[ -f "results/runs/${tag}.csv" ]]; then echo "skip $tag"; continue; fi
  echo "=== $(date '+%F %T') start $tag"
  $PY scripts/run_experiment.py --protocol "$proto" --frontend "$fe" --model "$model" --aug "$aug" \
      > "experiments/logs/${tag}.log" 2>&1
  echo "=== $(date '+%F %T') done  $tag (exit $?)"; tail -3 "experiments/logs/${tag}.log"
done < "$1"
