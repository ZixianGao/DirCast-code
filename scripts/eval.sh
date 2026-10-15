#!/usr/bin/env bash
# Evaluate a trained DirCast on the full test split: free-running streaming decode with the final decoding
# (rate-controlled thresholds, speech rate starts at 0, end threshold 0.3), then directing metrics and the
# Proact-VL commentary metrics.
#   bash scripts/eval.sh <ckpt.pt> <out_dir> [num_gpus=4]
set -euo pipefail
CKPT="${1:?usage: scripts/eval.sh <ckpt.pt> <out_dir> [num_gpus]}"; OUT="${2:?}"; NGPU="${3:-4}"
mkdir -p "$OUT"
torchrun --nproc_per_node "$NGPU" -m dircast.evaluate --ckpt "$CKPT" --out "$OUT" --modes free \
  --rate-control --target-speak 0.85 --speak-rate-init 0 --end-threshold 0.3
# directing: Top-1 / Any-visible / kill coverage against the human director, plus rule baselines
python -m dircast.director_eval --pred "$OUT/director_free.jsonl" | tee "$OUT/director.txt"
python -m dircast.director_eval_quality --pred "dircast=$OUT/director_free.jsonl" | tee -a "$OUT/director.txt"
# commentary: F1 / TimeDiff (and CC if a judge endpoint is configured), official Proact-VL scripts
bash "$(dirname "$0")/eval_commentary.sh" dircast "$OUT/commentary_free.jsonl" "$OUT/labels_asr_test.jsonl" "$OUT/commentary"
