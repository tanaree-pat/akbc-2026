#!/bin/bash
# Reproduce personHasCityOfDeath: vote>=4 of 4 prompt variants
# Variants: recent_aware, meta_antidefault, city_precision, recent_precise
# Assembly: entity gets prediction only if all 4 variants agree on the same city
# (see scripts/assemble_submission.py for the final voting logic)

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT_DIR=data/personDeath_runs

mkdir -p "$OUTPUT_DIR"

echo "=== personHasCityOfDeath: 4-way vote (single shot per variant) ==="
for style in recent_aware meta_antidefault city_precision recent_precise; do
    echo "--- Variant: $style ---"
    cp "$INPUT" "$OUTPUT_DIR/${style}.jsonl"
    $PY run_relation.py \
        --relation personHasCityOfDeath \
        --provider openrouter \
        --model qwen/qwen3.6-27b \
        --input "$INPUT" \
        --output "$OUTPUT_DIR/${style}.jsonl" \
        --prompt-style "${style}" \
        --log-name personDeath_${style}
    echo "Variant $style complete."
done

echo ""
echo "All 4 variants complete."
echo "Now assemble with: python scripts/assemble_submission.py --input $INPUT --output data/predictions_v6.jsonl"
