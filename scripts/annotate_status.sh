#!/usr/bin/env bash
# Usage: bash scripts/annotate_status.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re, time, subprocess

root = Path(".")
# Prefer live nohup logs; skip failed tiny stubs when a larger sibling exists.
cands = sorted((root / "logs").glob("annotate*.log"), key=lambda p: p.stat().st_mtime)
if not cands:
    print("no annotate logs under logs/")
    raise SystemExit(1)
# Prefer logs that contain progress, else newest
log = cands[-1]
for c in reversed(cands):
    t = c.read_text(errors="replace")
    if "s/it" in t or "Resuming" in t or "Final dataset" in t:
        log = c
        break
alive = subprocess.call(["pgrep", "-f", "src.annotate_data"], stdout=subprocess.DEVNULL) == 0
detached = False
try:
    out = subprocess.check_output(["pgrep", "-f", "server_run.sh annotate"], text=True).split()
    for pid in out:
        pp = Path(f"/proc/{pid}/status").read_text()
        for line in pp.splitlines():
            if line.startswith("PPid:"):
                detached = line.split()[1] == "1"
                break
except Exception:
    pass
parts = log.read_text(errors="replace").replace("\r", "\n").splitlines()
hits = [l for l in parts if "s/it" in l and "pass_at" in l]
line = hits[-1] if hits else ""
print(f"alive: {'yes' if alive else 'NO'}  detached_from_IDE: {'yes' if detached else 'no/unknown'}")
print(f"log:   {log}")
if not line:
    print("progress: (no step yet — still loading / first batch)")
else:
    m = re.search(r"(\d+)/(\d+).*?([\d.]+)s/it", line)
    if not m:
        print("progress: parse failed")
        print(line[-200:])
    else:
        done, total, sit = int(m.group(1)), int(m.group(2)), float(m.group(3))
        rem_s = max(0, total - done) * sit
        eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + rem_s))
        pct = 100.0 * done / total if total else 0
        print(f"progress: {done}/{total} ({pct:.2f}%)  {sit:.1f} s/it")
        print(f"remain:   {rem_s/3600:.1f} h  (~{rem_s/86400:.2f} d)")
        print(f"ETA~:     {eta}  (wall-clock estimate)")
        # postfix metrics if present
        mm = re.search(r"pass_at_1=([0-9.]+%).*?vot_acc=([0-9.]+%)", line)
        if mm:
            print(f"online:   pass@1={mm.group(1)}  vot={mm.group(2)}")

train = root / "latent-data/coconut/train"
shards = len(list(train.glob("*.safetensors"))) if train.is_dir() else 0
done_idx = root / "latent-data/coconut/train_done_idx.txt"
n_idx = sum(1 for _ in done_idx.open()) if done_idx.exists() else 0
print(f"train:   shards={shards}  done_idx_file={n_idx}")
print(f"valid-4: {'yes' if (root/'latent-data/coconut/valid-4').exists() else 'no'}")
PY
