#!/usr/bin/env python3
"""Reproduce personHasCityOfDeath: vote>=4 of 4 prompt variants.

Variants: recent_aware, meta_antidefault, city_precision, recent_precise.
Assembly: entity gets a prediction only if all 4 variants agree on the same city
(see scripts/assemble_submission.py for the final voting logic).

Standalone Python equivalent of the old scripts/run_personDeath.sh -- drives
run_relation.py's main() directly for each variant (same import-and-call
convention as scripts/run_hasArea.py and scripts/meta_prompt.py) instead of
looping over it in bash.

Usage (run from the repository root):
  export OPENROUTER_API_KEY="sk-or-..."
  python scripts/run_personDeath.py --input data/test.jsonl
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_relation

VARIANTS = ["recent_aware", "meta_antidefault", "city_precision", "recent_precise"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/val.jsonl",
                    help="pass data/test.jsonl to reproduce the submission")
    ap.add_argument("--output-dir", default="data/personDeath_runs")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY in your environment (see README)")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== personHasCityOfDeath: 4-way vote (single shot per variant) ===", flush=True)
    for style in VARIANTS:
        print(f"--- Variant: {style} ---", flush=True)
        out_path = os.path.join(args.output_dir, f"{style}.jsonl")
        shutil.copyfile(args.input, out_path)
        sys.argv = [
            "run_relation.py",
            "--relation", "personHasCityOfDeath",
            "--provider", "openrouter",
            "--model", "qwen/qwen3.6-27b",
            "--input", args.input,
            "--output", out_path,
            "--prompt-style", style,
            "--log-name", f"personDeath_{style}",
        ]
        run_relation.main()
        print(f"Variant {style} complete.", flush=True)

    print("\nAll 4 variants complete.")
    print(f"Now assemble with: python scripts/assemble_submission.py --input {args.input} "
          f"--output data/predictions.jsonl")


if __name__ == "__main__":
    main()
