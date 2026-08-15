#!/bin/bash
# Reproduce countryLandBordersCountry: single exhaustive prompt, single-shot

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT=${2:-data/countryBorders_out.jsonl}

cp "$INPUT" "$OUTPUT"

echo "=== countryLandBordersCountry: base exhaustive prompt, single-shot ==="
$PY run_relation.py \
    --relation countryLandBordersCountry \
    --provider openrouter \
    --model qwen/qwen3.6-27b \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --log-name countryBorders

echo "countryLandBordersCountry done. Output: $OUTPUT"
