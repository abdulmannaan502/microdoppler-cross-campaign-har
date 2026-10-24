"""Cross-platform queue runner (used on Kaggle; equivalent to run_queue.sh).
Each queue line: protocol frontend model aug [extra run_experiment args, e.g. --tent --tag NAME].
Conditions whose result CSV exists are skipped, so an interrupted queue can simply be restarted.

usage: python scripts/run_queue.py QUEUE_FILE [--gpu 0]
"""
import argparse, os, shlex, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--reverse", action="store_true", help="process the queue from the end (helper worker)")
    ap.add_argument("--skip-started", action="store_true", help="also skip conditions that already have a log")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    (ROOT / "experiments" / "logs").mkdir(parents=True, exist_ok=True)
    lines = Path(a.queue).read_text().splitlines()
    for line in (reversed(lines) if a.reverse else lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = shlex.split(line)
        if tok[0] == "enroll":  # enroll FRONTEND MODEL [extra]
            fe, model, extra = tok[1], tok[2], tok[3:]
            tag = f"enroll__{fe}__{model}"
            cmd = [sys.executable, "scripts/run_enroll.py", "--frontend", fe, "--model", model, *extra]
        else:
            proto, fe, model, aug, extra = tok[0], tok[1], tok[2], tok[3], tok[4:]
            tag = extra[extra.index("--tag") + 1] if "--tag" in extra else f"{proto}__{fe}__{model}__{aug}"
            cmd = [sys.executable, "scripts/run_experiment.py", "--protocol", proto, "--frontend", fe,
                   "--model", model, "--aug", aug, *extra]
        if (ROOT / "results" / "runs" / f"{tag}.csv").exists():
            print("skip", tag, flush=True)
            continue
        if a.skip_started and (ROOT / "experiments" / "logs" / f"{tag}.log").exists():
            print("skip (started elsewhere)", tag, flush=True)
            continue
        t0 = time.time()
        print(f"=== {time.strftime('%F %T')} start {tag} (gpu {a.gpu})", flush=True)
        with open(ROOT / "experiments" / "logs" / f"{tag}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"=== {time.strftime('%F %T')} done  {tag} rc={rc} ({(time.time() - t0) / 60:.1f} min)", flush=True)
