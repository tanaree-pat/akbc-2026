#!/usr/bin/env python3
"""Reproduce hasCapacity: country_tier prompt + confidence-escalated SC (5->9, conf>=0.80).

Standalone Python equivalent of the old scripts/run_hasCapacity.sh -- drives
run_relation.py's main() directly (same import-and-call convention as
scripts/run_hasArea.py and scripts/meta_prompt.py) instead of shelling out.

Usage (run from the repository root):
  export OPENROUTER_API_KEY="sk-or-..."
  python scripts/run_hasCapacity.py --input data/test.jsonl --output data/hasCapacity_out.jsonl
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_relation


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/val.jsonl",
                    help="pass data/test.jsonl to reproduce the submission")
    ap.add_argument("--output", default="data/hasCapacity_out.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY in your environment (see README)")

    shutil.copyfile(args.input, args.output)

    print("=== hasCapacity: country_tier prompt, SC 5->9, conf>=0.80 ===", flush=True)
    sys.argv = [
        "run_relation.py",
        "--relation", "hasCapacity",
        "--provider", "openrouter",
        "--model", "qwen/qwen3.6-27b",
        "--input", args.input,
        "--output", args.output,
        "--prompt-style", "country_tier",
        "--samples", "5", "--samples-escalate", "9", "--confidence-min", "0.80",
        "--log-name", "hasCapacity_sc",
    ]
    run_relation.main()

    print(f"hasCapacity done. Output: {args.output}")


if __name__ == "__main__":
    main()
