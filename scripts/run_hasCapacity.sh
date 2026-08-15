#!/bin/bash
# Reproduce hasCapacity: country_tier prompt + confidence-escalated SC (5->9, conf>=0.80)

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT=${2:-data/hasCapacity_out.jsonl}

cp "$INPUT" "$OUTPUT"

echo "=== hasCapacity: country_tier prompt, SC 5->9, conf>=0.80 ==="
$PY run_relation.py \
    --relation hasCapacity \
    --provider openrouter \
    --model qwen/qwen3.6-27b \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --prompt-style country_tier \
    --samples 5 --samples-escalate 9 --confidence-min 0.80 \
    --log-name hasCapacity_sc

echo "hasCapacity done. Output: $OUTPUT"
