#!/usr/bin/env bash
# Commentary metrics with the official Proact-VL evaluation scripts
# (github.com/microsoft/AnthropomorphicIntelligence, Proact-VL/evaluation/gaming).
#   bash scripts/eval_commentary.sh <model_id> <prediction.jsonl> <asr_reference.jsonl> <out_dir>
# Environment:
#   PROACTVL_EVAL   path to Proact-VL/evaluation/gaming (required)
#   SKIP_JUDGE=1    only F1 / TimeDiff (no API calls)
#   CAPTIONS        offline GPT-4o captions used as the CC opponent (required for CC)
#   OPENAI_API_KEY, OPENAI_BASE_URL, JUDGE_MODEL   judge endpoint for CC / LLM scores (we used gpt-5)
set -euo pipefail
MID="$1"; PRED="$(realpath "$2")"; ASR="$(realpath "$3")"; OUT="$4"; mkdir -p "$OUT"; OUT="$(realpath "$OUT")"
: "${PROACTVL_EVAL:?set PROACTVL_EVAL to Proact-VL/evaluation/gaming}"
cd "$PROACTVL_EVAL"
python f1_timediff.py "$PRED" --output "$OUT/f1_timediff.json" --alpha 0.2 | tail -12
if [[ "${SKIP_JUDGE:-0}" != "1" ]]; then
  : "${CAPTIONS:?set CAPTIONS to the GPT-4o offline captions for the CC opponent}"
  python llm_judge.py --model_id "$MID" --prediction_jsonl "$PRED" --asr_jsonl "$ASR" \
    --baseline_id gpt-4o --baseline_jsonl "$CAPTIONS" --output_dir "$OUT/judge" --num_workers "${JUDGE_WORKERS:-8}" | tail -3
fi
if [[ "${LLM_SCORE:-0}" == "1" ]]; then
  python llm_score.py --model_id "$MID" --prediction_jsonl "$PRED" --asr_jsonl "$ASR" \
    --output_dir "$OUT/llm_score" --num_workers "${JUDGE_WORKERS:-8}" | tail -5
fi
