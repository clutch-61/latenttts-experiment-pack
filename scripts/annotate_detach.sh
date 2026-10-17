#!/usr/bin/env bash
# Detach annotate from Cursor/IDE: survives logout and editor disconnect.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate latenttts

# Refresh resume index from shards on disk
python3 - <<'PY'
from pathlib import Path
from safetensors import safe_open
root = Path("latent-data/coconut/train")
idxs = set()
if root.is_dir():
    for f in root.glob("*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.endswith(".input_ids"):
                    idxs.add(int(k.split(".")[0]))
Path("latent-data/coconut/train_done_idx.txt").write_text(
    "\n".join(str(i) for i in sorted(idxs)) + ("\n" if idxs else "")
)
print(f"refreshed done_idx={len(idxs)} shards={len(list(root.glob('*.safetensors'))) if root.is_dir() else 0}")
PY

mkdir -p logs
LOG="logs/annotate_nohup_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTHONUNBUFFERED=1

# New session, ignore hangup; not a child of cursor-server
nohup setsid bash scripts/server_run.sh annotate >>"$LOG" 2>&1 < /dev/null &
echo "started pid=$!"
echo "log=$ROOT/$LOG"
sleep 2
pgrep -af "src.annotate_data|server_run.sh annotate" | grep -v grep || echo "WARN: not visible yet"
