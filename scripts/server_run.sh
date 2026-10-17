#!/usr/bin/env bash
# Server entry scripts for LatentTTS baseline + Phase-1 order preference.
# Do NOT download weights here. Place checkpoints under ./checkpoints/ before running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== env ==="
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo "=== order_pref self-check (no weights) ==="
python -m src.order_pref.check_order_pref

need() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "MISSING: $p"
    echo "Put HF weights on the server (not on the laptop), e.g.:"
    echo "  huggingface-cli download ModalityDance/latent-tts-coconut --local-dir checkpoints/coconut"
    echo "  huggingface-cli download ModalityDance/latent-tts-rm --local-dir checkpoints/latentRM"
    exit 1
  fi
}

CMD="${1:-check}"
case "$CMD" in
  check)
    echo "OK: self-check passed. Next: bash scripts/server_run.sh annotate|train_baseline|train_pref|eval_tts|eval_order"
    ;;
  annotate)
    need checkpoints/coconut
    mkdir -p latent-data/coconut
    bash run_annotation.sh
    ;;
  train_baseline)
    need checkpoints/coconut
    need latent-data/coconut/train
    accelerate launch -m src.train training_args/train_baseline_ce.yaml
    ;;
  train_pref)
    need checkpoints/coconut
    need latent-data/coconut/train
    # smaller batches because each step also forwards K order-negatives
    accelerate launch -m src.train_order_pref training_args/train_order_pref.yaml
    ;;
  eval_tts)
    need checkpoints/coconut
    need checkpoints/latentRM
    bash run_tts_with_rm.sh
    ;;
  eval_order)
    PRM="${2:-outputs/latentrm_order_pref/best}"
    need "$PRM"
    need latent-data/coconut/valid-4
    python -m src.eval_order_pref --prm_id="$PRM" --data_dir=latent-data/coconut/valid-4
    ;;
  *)
    echo "usage: $0 {check|annotate|train_baseline|train_pref|eval_tts|eval_order [prm_dir]}"
    exit 2
    ;;
esac
