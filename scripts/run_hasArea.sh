#!/bin/bash
# Reproduce hasArea: 6-run cluster-median ensemble
# Runs 1-5: base prompt with confidence-escalated SC (5->9, conf>=0.80)
# Run 6: native_lang_anchored prompt with confidence-escalated SC (5->9, conf>=0.80)
# Final assembly: cluster-median across all 6 runs (see scripts/assemble_submission.py)

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT_DIR=data/hasArea_runs

mkdir -p "$OUTPUT_DIR"

echo "=== hasArea run 1-5: base prompt, SC 5->9, conf>=0.80 ==="
for i in 1 2 3 4 5; do
    echo "--- Run $i ---"
    cp "$INPUT" "$OUTPUT_DIR/run${i}.jsonl"
    $PY run_relation.py \
        --relation hasArea \
        --provider openrouter \
        --model qwen/qwen3.6-27b \
        --input "$INPUT" \
        --output "$OUTPUT_DIR/run${i}.jsonl" \
        --samples 5 --samples-escalate 9 --confidence-min 0.80 \
        --log-name hasArea_run${i}
    echo "Run $i complete."
done

echo "=== hasArea run 6: native_lang_anchored prompt, SC 5->9, conf>=0.80 ==="
cp "$INPUT" "$OUTPUT_DIR/run6_anchored.jsonl"
$PY run_relation.py \
    --relation hasArea \
    --provider openrouter \
    --model qwen/qwen3.6-27b \
    --input "$INPUT" \
    --output "$OUTPUT_DIR/run6_anchored.jsonl" \
    --prompt-style native_lang_anchored \
    --samples 5 --samples-escalate 9 --confidence-min 0.80 \
    --log-name hasArea_run6_anchored
echo "Run 6 complete."

echo ""
echo "All 6 runs complete."
echo "Now assemble with: python scripts/assemble_submission.py --input $INPUT --output data/predictions_v6.jsonl"
