#!/usr/bin/env python3
"""Meta-prompting (technique #1) + thinking-trace review (technique #2).

This is the "error-driven prompt refinement" technique described in the paper
(Section: Methods; also called meta-prompting in early drafts / DOCUMENTATION.md
Phase 26). It is a form of automatic prompt optimization (cf. Pryzant et al., APO):
the model is shown its own current prompt, a sample of its real failure cases
(prediction vs. gold), and a few raw reasoning traces, then asked to diagnose the
dominant failure pattern and propose improved prompt templates.

The `meta_precision`, `meta_verify`, and `meta_antidefault` prompt variants used in
the final companyTradesAtStockExchange and personHasCityOfDeath ensembles (see
models/prompts.py) were produced this way.

Candidates are saved to meta_prompts/{relation}.txt for human review BEFORE testing
-- this script never auto-injects a candidate into the pipeline. A human reads the
output, decides whether to adopt/adapt it, pastes the chosen prompt into
models/prompts.py under PROMPT_VARIANTS, and re-evaluates on val with run_relation.py
like any other prompt variant.

Compliance: this is prompt engineering only. No fine-tuning, no weight/gradient access.

Usage (run from the repository root):
  export OPENROUTER_API_KEY="sk-or-..."
  python scripts/meta_prompt.py --relation personHasCityOfDeath \\
      --pred-file data/val_predictions.jsonl \\
      --base-style recent_aware \\
      --log logs/personDeath_recent_aware.jsonl
"""
import argparse
import json
import os
import sys
from openai import OpenAI

# Allow running as `python scripts/meta_prompt.py` from the repo root: put the repo
# root (the parent of scripts/) on the path so models/evaluate import, matching the
# convention used by scripts/run_hasArea.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prompts import PROMPTS, PROMPT_VARIANTS
from evaluate import try_parse_number

NUMERIC = {"hasArea", "hasCapacity"}


def gold_map(gold_file, relation):
    m = {}
    for l in open(gold_file):
        r = json.loads(l)
        if r["Relation"] == relation:
            m[r["SubjectEntity"]] = r["ObjectEntities"]
    return m


def is_correct(pred, gold, relation):
    """gold is List[List[str]] (aliases). pred is List[str]."""
    if relation in NUMERIC:
        if not pred:
            return not gold
        pv = try_parse_number(pred[0])
        if pv is None:
            return False
        for g in gold:
            gv = try_parse_number(g[0] if isinstance(g, list) else g)
            if gv is not None and abs(pv - gv) <= 0.05 * abs(gv):
                return True
        return False
    # text: flatten gold aliases, case-insensitive membership (partial credit ignored here)
    gset = {str(a).lower() for grp in gold for a in (grp if isinstance(grp, list) else [grp])}
    if not gold:
        return not pred          # correct-empty
    if not pred:
        return False             # missed
    return any(p.lower() in gset for p in pred)


def collect_failures(pred_file, gold, relation, limit=15):
    fails = []
    for l in open(pred_file):
        r = json.loads(l)
        if r["Relation"] != relation:
            continue
        s = r["SubjectEntity"]
        pred = r["ObjectEntities"]
        g = gold.get(s, [])
        if not is_correct(pred, g, relation):
            gflat = [grp[0] if isinstance(grp, list) else grp for grp in g] or ["(empty)"]
            fails.append({"subject": s, "pred": pred or ["(empty)"], "gold": gflat})
    return fails[:limit]


def sample_traces(log_file, n=4, max_chars=1200):
    """Pull a few raw thinking traces to show the model HOW it reasoned."""
    if not log_file or not os.path.exists(log_file):
        return []
    rows = [json.loads(l) for l in open(log_file)]
    out = []
    for r in rows[:n]:
        raw = r.get("raw", "")
        out.append({"subject": r.get("subject", "?"), "trace": raw[:max_chars]})
    return out


def build_meta_prompt(relation, current_prompt, fails, traces):
    lines = []
    lines.append(f"You are an expert prompt engineer improving a closed-book knowledge-extraction "
                 f"prompt for the relation '{relation}'. The model must answer from its own "
                 f"knowledge only (no web/RAG) and output a JSON array.\n")
    lines.append("=== CURRENT PROMPT ===")
    lines.append(current_prompt)
    lines.append("\n=== FAILURE CASES (model prediction vs correct gold) ===")
    for f in fails:
        lines.append(f"- {f['subject']}: predicted {f['pred']}  |  correct {f['gold']}")
    if traces:
        lines.append("\n=== SAMPLE REASONING TRACES (how the model thought) ===")
        for t in traces:
            lines.append(f"[{t['subject']}]\n{t['trace']}\n")
    lines.append(
        "\n=== YOUR TASK ===\n"
        "1. In 3-4 sentences, diagnose the DOMINANT failure pattern (e.g. over-predicting when it "
        "should abstain, wrong granularity, hallucinating, missing recent facts).\n"
        "2. Then write TWO improved prompt templates that target that failure pattern. Keep the "
        "{subject} placeholder, keep the JSON-array output instruction, and stay closed-book. "
        "Do not add facts to the prompt -- only change instructions/reasoning structure.\n"
        "Format exactly as:\n"
        "DIAGNOSIS: <your diagnosis>\n"
        "PROMPT_A:\n<template A>\n\nPROMPT_B:\n<template B>\n"
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relation", required=True)
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--gold", default="data/val.jsonl")
    ap.add_argument("--base-style", default="simple",
                    help="which current prompt to show: simple or a PROMPT_VARIANTS key")
    ap.add_argument("--log", default="", help="run log jsonl with raw traces (optional)")
    ap.add_argument("--model", default="qwen/qwen3.6-27b")
    ap.add_argument("--max-fails", type=int, default=15)
    args = ap.parse_args()

    current = (PROMPT_VARIANTS.get(args.relation, {}).get(args.base_style)
               or PROMPTS[args.relation])
    gold = gold_map(args.gold, args.relation)
    fails = collect_failures(args.pred_file, gold, args.relation, limit=args.max_fails)
    traces = sample_traces(args.log)
    meta = build_meta_prompt(args.relation, current, fails, traces)

    print(f"[meta-prompting {args.relation}: {len(fails)} failure cases, {len(traces)} traces]", flush=True)
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY"))
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": meta}],
        max_tokens=4000, temperature=0.4,
    )
    out = resp.choices[0].message.content or ""
    reasoning = getattr(resp.choices[0].message, "reasoning", None)
    if reasoning and "</think>" not in out:
        out = out.split("</think>")[-1]

    os.makedirs("meta_prompts", exist_ok=True)
    path = f"meta_prompts/{args.relation}.txt"
    with open(path, "w") as f:
        f.write(out)
    print("\n" + out)
    print(f"\n[saved candidates -> {path}]  Review, then paste the best into models/prompts.py "
          f"and test with run_relation.py.", flush=True)


if __name__ == "__main__":
    main()
