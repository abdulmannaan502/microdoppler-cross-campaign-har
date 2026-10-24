#!/usr/bin/env bash
# Remote runner: install missing deps, run 4 queue workers in parallel on one GPU.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/venv/main/bin/python}
export TORCH_HOME="$PWD/remote/torch_home"
$PY -m pip install -q scikit-learn pandas pyyaml joblib scipy tabulate pytest matplotlib >/dev/null 2>&1 || true
$PY -c "import torch,torchvision,sklearn,pandas,numpy;print('torch',torch.__version__,'tv',torchvision.__version__,'np',numpy.__version__,'sk',sklearn.__version__,'pd',pandas.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" | tee remote/env.txt
nvidia-smi >> remote/env.txt
$PY -m pytest -q tests >> remote/env.txt 2>&1 || true
tail -3 remote/env.txt
for w in 0 1 2 3; do
  TORCH_HOME="$TORCH_HOME" nohup $PY scripts/run_queue.py experiments/configs/remote_w$w.txt --gpu 0 > remote/worker$w.out 2>&1 &
done
echo "launched 4 workers"
