#!/usr/bin/env python3
"""Reproduce awardWonBy: union of two recall passes, each with a per-name confidence vote.

Passes: alphabetical (sweep A-Z) and year_sweep (year-by-year).
Per pass: SC x5 samples per entity; keep a name if >=2/5 (40%) samples include it.
Assembly: union = alphabetical names + year_sweep names not already present
(see assemble_awardWonBy in scripts/assemble_submission.py).

Standalone Python equivalent of the old scripts/run_awardWonBy.sh -- drives
run_relation.py's main() directly for each pass (same import-and-call
convention as scripts/run_hasArea.py and scripts/meta_prompt.py) instead of
looping over it in bash.

Usage (run from the repository root):
  export OPENROUTER_API_KEY="sk-or-..."
  python scripts/run_awardWonBy.py --input data/test.jsonl
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_relation

PASSES = ["alphabetical", "year_sweep"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/val.jsonl",
                    help="pass data/test.jsonl to reproduce the submission")
    ap.add_argument("--output-dir", default="data/awardWonBy_runs")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY in your environment (see README)")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== awardWonBy: alphabetical + year_sweep passes "
          "(each per-name vote SC x5, conf>=0.40) ===", flush=True)
    for style in PASSES:
        print(f"--- Pass: {style} ---", flush=True)
        out_path = os.path.join(args.output_dir, f"{style}.jsonl")
        shutil.copyfile(args.input, out_path)
        sys.argv = [
            "run_relation.py",
            "--relation", "awardWonBy",
            "--provider", "openrouter",
            "--model", "qwen/qwen3.6-27b",
            "--input", args.input,
            "--output", out_path,
            "--prompt-style", style,
            "--award-samples", "5", "--confidence-min", "0.40",
            "--log-name", f"awardWonBy_{style}",
        ]
        run_relation.main()
        print(f"Pass {style} complete.", flush=True)

    print("\nBoth passes complete.")
    print(f"Now assemble with: python scripts/assemble_submission.py --input {args.input} "
          f"--output data/predictions.jsonl")


if __name__ == "__main__":
    main()
