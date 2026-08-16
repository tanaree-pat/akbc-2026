#!/usr/bin/env python3
"""
assemble_submission.py
Assemble per-relation output files into a single submission JSONL.

Per-relation assembly logic:
  hasArea          — cluster-median across the 6 runs from run_hasArea.py
  hasCapacity      — single run (country_tier + SC), copy directly
  personDeath      — vote>=4 of 4 variants (all 4 must agree on same city)
  countryBorders   — single run, copy directly
  companyTrades    — vote>=3 of 5 variants (exchange kept if >=3 predict it)
  awardWonBy       — union of alphabetical + year_sweep passes (each per-name SC vote)

Usage:
    python scripts/assemble_submission.py \\
        --input data/test.jsonl \\
        --output data/predictions_v6.jsonl

All per-relation files must exist before running this script.
Run scripts/run_*.sh first to generate them.
"""

import argparse
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization (same logic as evaluate.py)
# ---------------------------------------------------------------------------
APOSTROPHE_LIKE = set("'''ʻʼʹ`´")
ASCII_SYMBOLS = set("+$<=>|~^")


def normalize_string(s: str) -> str:
    s = "".join(c for c in s.strip() if c not in APOSTROPHE_LIKE)
    s = unicodedata.normalize("NFKD", s).casefold()
    out = []
    for c in s:
        if c in APOSTROPHE_LIKE or unicodedata.combining(c):
            continue
        out.append(" " if c in ASCII_SYMBOLS or unicodedata.category(c).startswith("P") else c)
    return " ".join("".join(out).split())


# ---------------------------------------------------------------------------
# Numeric helpers (cluster-median)
# ---------------------------------------------------------------------------
def try_parse_number(value: str):
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def cluster_median(values, tol=0.05):
    """Find densest cluster within tol relative tolerance, return its median."""
    if not values:
        return None
    best = []
    for v in values:
        scale = max(abs(v), 1e-9)
        supporters = [u for u in values if abs(u - v) <= tol * scale]
        if len(supporters) > len(best):
            best = supporters
    med = statistics.median(best)
    return med


def format_number(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return str(round(v, 6))


# ---------------------------------------------------------------------------
# Load a jsonl file as {(SubjectEntity, Relation): ObjectEntities}
# ---------------------------------------------------------------------------
def load_file(path: str):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows[(r["SubjectEntity"], r["Relation"])] = r.get("ObjectEntities", [])
    return rows


# ---------------------------------------------------------------------------
# Assemblers per relation
# ---------------------------------------------------------------------------

def assemble_hasArea(input_rows, data_dir: str):
    """Cluster-median across the 6 runs produced by run_hasArea.py."""
    run_files = [
        f"{data_dir}/hasArea_runs/run1.jsonl",
        f"{data_dir}/hasArea_runs/run2.jsonl",
        f"{data_dir}/hasArea_runs/run3.jsonl",
        f"{data_dir}/hasArea_runs/run4.jsonl",
        f"{data_dir}/hasArea_runs/run5.jsonl",
        f"{data_dir}/hasArea_runs/run6_anchored.jsonl",
    ]
    runs = []
    for f in run_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing hasArea run file: {f} (run run_hasArea.py first)")
        runs.append(load_file(f))

    result = {}
    for row in input_rows:
        if row["Relation"] != "hasArea":
            continue
        entity = row["SubjectEntity"]
        all_nums = []
        for run in runs:
            preds = run.get((entity, "hasArea"), [])
            if preds:
                v = try_parse_number(preds[0])
                if v is not None:
                    all_nums.append(v)
        if all_nums:
            med = cluster_median(all_nums)
            result[entity] = [format_number(med)]
        else:
            result[entity] = []
    return result


def assemble_personDeath(input_rows, data_dir: str):
    """Vote>=4 of 4: entity gets prediction only if all 4 variants agree on same city."""
    variant_files = [
        f"{data_dir}/personDeath_runs/recent_aware.jsonl",
        f"{data_dir}/personDeath_runs/meta_antidefault.jsonl",
        f"{data_dir}/personDeath_runs/city_precision.jsonl",
        f"{data_dir}/personDeath_runs/recent_precise.jsonl",
    ]
    variants = []
    for f in variant_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing personDeath variant file: {f}")
        variants.append(load_file(f))

    result = {}
    for row in input_rows:
        if row["Relation"] != "personHasCityOfDeath":
            continue
        entity = row["SubjectEntity"]
        cities = []
        for v in variants:
            preds = v.get((entity, "personHasCityOfDeath"), [])
            cities.append(normalize_string(preds[0]) if preds else "")

        # All 4 must agree on the same non-empty city
        non_empty = [c for c in cities if c]
        if len(non_empty) == 4 and len(set(non_empty)) == 1:
            # All 4 agree; use the display form from the first variant
            preds0 = variants[0].get((entity, "personHasCityOfDeath"), [])
            result[entity] = [preds0[0]] if preds0 else []
        else:
            result[entity] = []
    return result


def assemble_companyTrades(input_rows, data_dir: str):
    """Vote>=3 of 5: keep an exchange if >=3 of 5 variants predict it."""
    variant_files = [
        f"{data_dir}/companyTrades_runs/simple.jsonl",
        f"{data_dir}/companyTrades_runs/listed_check.jsonl",
        f"{data_dir}/companyTrades_runs/meta_precision.jsonl",
        f"{data_dir}/companyTrades_runs/meta_verify.jsonl",
        f"{data_dir}/companyTrades_runs/anti_confusion.jsonl",
    ]
    variants = []
    for f in variant_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing companyTrades variant file: {f}")
        variants.append(load_file(f))

    result = {}
    for row in input_rows:
        if row["Relation"] != "companyTradesAtStockExchange":
            continue
        entity = row["SubjectEntity"]
        # Count votes per normalized exchange name
        vote_count: Counter = Counter()
        canonical = {}
        for v in variants:
            preds = v.get((entity, "companyTradesAtStockExchange"), [])
            for exchange in preds:
                norm = normalize_string(exchange)
                vote_count[norm] += 1
                if norm not in canonical:
                    canonical[norm] = exchange
        kept = [canonical[norm] for norm, cnt in vote_count.items() if cnt >= 3]
        result[entity] = kept
    return result


def assemble_awardWonBy(input_rows, data_dir: str):
    """Union of the alphabetical and year_sweep passes: keep all alphabetical names,
    then add year_sweep names not already present (normalized, case-insensitive)."""
    alpha_file = f"{data_dir}/awardWonBy_runs/alphabetical.jsonl"
    sweep_file = f"{data_dir}/awardWonBy_runs/year_sweep.jsonl"
    for f in (alpha_file, sweep_file):
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing awardWonBy pass file: {f} (run run_awardWonBy.sh first)")
    alpha = load_file(alpha_file)
    sweep = load_file(sweep_file)

    result = {}
    for row in input_rows:
        if row["Relation"] != "awardWonBy":
            continue
        entity = row["SubjectEntity"]
        a = alpha.get((entity, "awardWonBy"), [])
        s = sweep.get((entity, "awardWonBy"), [])
        seen = {normalize_string(x) for x in a}
        result[entity] = list(a) + [x for x in s if normalize_string(x) not in seen]
    return result


def assemble_single(input_rows, relation: str, file_path: str):
    """Single run: copy ObjectEntities directly."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    src = load_file(file_path)
    result = {}
    for row in input_rows:
        if row["Relation"] != relation:
            continue
        entity = row["SubjectEntity"]
        result[entity] = src.get((entity, relation), [])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assemble per-relation outputs into submission file")
    parser.add_argument("--input", default="data/test.jsonl",
                        help="Input JSONL with (SubjectEntity, Relation) pairs")
    parser.add_argument("--output", default="data/predictions_v6.jsonl",
                        help="Output assembled submission file")
    parser.add_argument("--data-dir", default="data",
                        help="Base data directory where per-relation run folders are")
    args = parser.parse_args()

    input_rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    print(f"Loaded {len(input_rows)} rows from {args.input}")

    # Assemble each relation
    print("Assembling hasArea (6-run cluster-median)...")
    hasArea = assemble_hasArea(input_rows, args.data_dir)

    print("Assembling hasCapacity (single run)...")
    hasCapacity = assemble_single(input_rows, "hasCapacity",
                                   f"{args.data_dir}/hasCapacity_out.jsonl")

    print("Assembling personHasCityOfDeath (vote>=4 of 4)...")
    personDeath = assemble_personDeath(input_rows, args.data_dir)

    print("Assembling countryLandBordersCountry (single run)...")
    countryBorders = assemble_single(input_rows, "countryLandBordersCountry",
                                      f"{args.data_dir}/countryBorders_out.jsonl")

    print("Assembling companyTradesAtStockExchange (vote>=3 of 5)...")
    companyTrades = assemble_companyTrades(input_rows, args.data_dir)

    print("Assembling awardWonBy (alphabetical + year_sweep union)...")
    awardWonBy = assemble_awardWonBy(input_rows, args.data_dir)

    # Combine into final output, preserving input row order
    lookup = {
        "hasArea": hasArea,
        "hasCapacity": hasCapacity,
        "personHasCityOfDeath": personDeath,
        "countryLandBordersCountry": countryBorders,
        "companyTradesAtStockExchange": companyTrades,
        "awardWonBy": awardWonBy,
    }

    out_rows = []
    for row in input_rows:
        rel = row["Relation"]
        entity = row["SubjectEntity"]
        preds = lookup.get(rel, {}).get(entity, [])
        out_rows.append({
            "SubjectEntity": entity,
            "Relation": rel,
            "ObjectEntities": preds,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_rows)} rows to {args.output}")
    from collections import Counter
    counts = Counter(r["Relation"] for r in out_rows)
    for rel, cnt in sorted(counts.items()):
        print(f"  {rel}: {cnt}")


if __name__ == "__main__":
    main()
