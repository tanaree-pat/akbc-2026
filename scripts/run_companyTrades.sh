#!/bin/bash
# Reproduce companyTradesAtStockExchange: vote>=3 of 5 prompt variants
# Variants: simple (base prompt), listed_check, meta_precision, meta_verify, anti_confusion
# Assembly: exchange kept if >=3 of 5 variants predict it
# (see scripts/assemble_submission.py for the final voting logic)

set -e

PY=/opt/anaconda3/envs/lm-kbc-2026/bin/python
eval "$(grep '^export OPENROUTER_API_KEY' ~/.zshrc)"; export OPENROUTER_API_KEY

INPUT=${1:-data/val.jsonl}   # pass data/test.jsonl as $1 for test
OUTPUT_DIR=data/companyTrades_runs

mkdir -p "$OUTPUT_DIR"

echo "=== companyTradesAtStockExchange: 5-way vote ==="
for style in simple listed_check meta_precision meta_verify anti_confusion; do
    echo "--- Variant: $style ---"
    cp "$INPUT" "$OUTPUT_DIR/${style}.jsonl"
    $PY run_relation.py \
        --relation companyTradesAtStockExchange \
        --provider openrouter \
        --model qwen/qwen3.6-27b \
        --input "$INPUT" \
        --output "$OUTPUT_DIR/${style}.jsonl" \
        --prompt-style "${style}" \
        --log-name companyTrades_${style}
    echo "Variant $style complete."
done

echo ""
echo "All 5 variants complete."
echo "Now assemble with: python scripts/assemble_submission.py --input $INPUT --output data/predictions_v6.jsonl"
