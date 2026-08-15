#!/bin/bash
# Reproduce awardWonBy: alphabetical sweep prompt + per-name confidence vote
# Method: SC x5 samples per entity; keep a name if >=2/5 (40%) samples include it
# Prompt style: "alphabetical" (sweeps memory A-Z for recipient names)

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT=${2:-data/awardWonBy_out.jsonl}

cp "$INPUT" "$OUTPUT"

echo "=== awardWonBy: alphabetical prompt + per-name confidence vote (SC x5, conf>=0.40) ==="
$PY run_relation.py \
    --relation awardWonBy \
    --provider openrouter \
    --model qwen/qwen3.6-27b \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --prompt-style alphabetical \
    --award-samples 5 --confidence-min 0.40 \
    --log-name awardWonBy

echo "awardWonBy done. Output: $OUTPUT"
