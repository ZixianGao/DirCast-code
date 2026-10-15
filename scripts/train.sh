#!/usr/bin/env bash
# Train DirCast with the recipe of the paper (Qwen3-VL-8B + LoRA r=64, 3 epochs).
#   bash scripts/train.sh <out_dir> [seed=0] [num_gpus=4]
# Needs DIRCAST_DATA (dataset root) or DIRCAST_INDEX (clip index with polished commentary).
set -euo pipefail
OUT="${1:?usage: scripts/train.sh <out_dir> [seed] [num_gpus]}"; SEED="${2:-0}"; NGPU="${3:-4}"
MODEL="${DIRCAST_BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
mkdir -p "$OUT"
torchrun --nproc_per_node "$NGPU" -m dircast.train \
  --model "$MODEL" --out "$OUT" --seed "$SEED" \
  --lo-side 128 --hi-side 288 --accum 4 \
  --use-events --ctx-drop 0.3 \
  --use-state --state-pointer --kill-feats \
  --prog-noise 0.06 --kill-weight 0.5
# base thresholds for the streaming decisions, calibrated on the validation split
torchrun --nproc_per_node "$NGPU" -m dircast.tune --ckpt "$OUT/ckpt_ep2.pt"
