#!/bin/bash
# Reproduce awardWonBy: union of two recall passes, each with a per-name confidence vote.
# Passes:  alphabetical (sweep A-Z) and year_sweep (year-by-year).
# Per pass: SC x5 samples per entity; keep a name if >=2/5 (40%) samples include it.
# Assembly: union = alphabetical names + year_sweep names not already present
#           (see assemble_awardWonBy in scripts/assemble_submission.py).

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT_DIR=data/awardWonBy_runs

mkdir -p "$OUTPUT_DIR"

echo "=== awardWonBy: alphabetical + year_sweep passes (each per-name vote SC x5, conf>=0.40) ==="
for style in alphabetical year_sweep; do
    echo "--- Pass: $style ---"
    cp "$INPUT" "$OUTPUT_DIR/${style}.jsonl"
    $PY run_relation.py \
        --relation awardWonBy \
        --provider openrouter \
        --model qwen/qwen3.6-27b \
        --input "$INPUT" \
        --output "$OUTPUT_DIR/${style}.jsonl" \
        --prompt-style "${style}" \
        --award-samples 5 --confidence-min 0.40 \
        --log-name awardWonBy_${style}
    echo "Pass $style complete."
done

echo ""
echo "Both passes complete."
echo "Now assemble with: python scripts/assemble_submission.py --input $INPUT --output data/predictions_v6.jsonl"
