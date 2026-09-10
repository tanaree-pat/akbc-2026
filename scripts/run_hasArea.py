#!/usr/bin/env python3
"""Generate all six hasArea runs in one file.

This produces the six per-run prediction files that make up the hasArea ensemble.
The cluster-median across the six runs is then done by scripts/assemble_submission.py
(so all final aggregation lives in one place, like the vote for the other relations).

Runs 1-5: base prompt, confidence-escalated self-consistency (5->9, cluster-median).
Run 6:    anchored native-language run with TWO-MODEL language identification:
            - qwen/qwen3.6-27b proposes the entity's primary local language + confidence;
            - if qwen confidence < 0.7, google/gemma-3-4b-it is also asked and the
              higher-confidence guess wins (fallback + max, not an average);
            - if the final confidence >= 0.7 the anchored prompt is translated into that
              language (the entity name stays in English so its identity is anchored),
              otherwise the English anchored prompt is used;
            - answered with the same confidence-escalated SC as runs 1-5.

Each run file stores, per entity, that run's single SC value (or []). The ensemble
(cluster-median across the six values) is applied later by assemble_submission.py.

Inference models: qwen 27B + gemma 4B = 31B, within the 32B budget.

Usage (run from the repository root, like the other scripts/ commands):
  python scripts/run_hasArea.py --input data/test.jsonl     # -> data/hasArea_runs/run{1..5}.jsonl + run6_anchored.jsonl
  python scripts/assemble_submission.py --input data/test.jsonl --output data/predictions.jsonl

Resumable & crash-safe: re-running skips (run, entity) pairs already in --rawlog and
keeps values already written to each run file.
"""
import os
import sys
import json
import argparse

# Allow running as `python scripts/run_hasArea.py` from the repo root: put the repo
# root (the parent of scripts/) on the path so run_relation / evaluate / models import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import read_jsonl_file, try_parse_number
from models.prompts import PROMPT_VARIANTS
from models.openrouter_model import OpenRouterModel
from run_relation import cluster_median, cluster_confidence

QWEN = "qwen/qwen3.6-27b"
GEMMA = "google/gemma-3-4b-it"
SAMPLES = 5             # self-consistency base batch
ESCALATE = 9            # escalate to this many when the batch disagrees
CONF_MIN = 0.80         # cluster-confidence threshold to escalate
LANG_THRESH = 0.70      # language-confidence threshold to issue the query in-language
ANSWER_MAX_TOKENS = 8000
ANCHORED = PROMPT_VARIANTS["hasArea"]["native_lang_anchored"]

# One qwen client (temperature flipped between 0.0 for language-ID/translation and 0.7
# for the sampled answers) and one small gemma client for the language-ID fallback.
_qwen = OpenRouterModel(model_name=QWEN, prompt_style="simple",
                        temperature=0.7, log_name="hasArea_qwen")
_gemma = OpenRouterModel(model_name=GEMMA, temperature=0.0, log_name="hasArea_gemma")


def _strip_think(resp: str) -> str:
    return resp.split("</think>")[-1].strip() if "</think>" in resp else resp.strip()


def _num_from(preds):
    return try_parse_number(preds[0]) if preds else None


def format_number(v):
    return str(int(v)) if v == int(v) else str(round(v, 6))


def escalated_sc(gen_one):
    """Confidence-escalated SC: draw SAMPLES; if the cluster confidence is below
    CONF_MIN, draw up to ESCALATE. Returns the cluster-median (float) or None.
    gen_one() must return a single parsed number (float) or None."""
    nums = []
    for _ in range(SAMPLES):
        v = gen_one()
        if v is not None:
            nums.append(v)
    conf = cluster_confidence(nums) if nums else 0.0
    if conf < CONF_MIN and ESCALATE > SAMPLES:
        for _ in range(ESCALATE - SAMPLES):
            v = gen_one()
            if v is not None:
                nums.append(v)
    return cluster_median(nums) if nums else None


# --- runs 1-5: base prompt -------------------------------------------------
def base_value(row, log):
    """One base-prompt SC run (temperature 0.7). Returns a float or None."""
    _qwen.temperature = 0.7
    v = escalated_sc(lambda: _num_from(_qwen.generate_predictions([row])[0]))
    log["used"] = "base"
    return v


# --- run 6: anchored native-language with two-model language ID -------------
def _parse_lang(resp):
    resp = _strip_think(resp)
    try:
        s = resp.rindex("{"); e = resp.rindex("}") + 1
        d = json.loads(resp[s:e])
        return str(d.get("language", "")).strip(), float(d.get("confidence", 0) or 0)
    except Exception:
        return "", 0.0


def identify_language(subject, log):
    p = (f'What is the primary local language of the place, country, or origin associated '
         f'with "{subject}"? Respond ONLY as JSON: '
         f'{{"language": "<language name>", "confidence": <0 to 1>}}')
    _qwen.temperature = 0.0
    lang, conf = _parse_lang(_qwen.call_openrouter(p, max_tokens=1500))
    log["lang_qwen"] = {"lang": lang, "conf": conf}
    if conf < LANG_THRESH:                 # unsure -> ask the small gemma too
        lg, cg = _parse_lang(_gemma.call_openrouter(p, max_tokens=300))
        log["lang_gemma"] = {"lang": lg, "conf": cg}
        if cg > conf:                      # fallback + max (not average)
            lang, conf = lg, cg
    return lang, conf


def translate_prompt(full, language, log):
    p = (f'Translate the following text into {language}. Keep every number, JSON array, and '
         f'the entity name EXACTLY as written. Output ONLY the translation, with no preamble '
         f'or commentary:\n\n{full}')
    _qwen.temperature = 0.0
    for _ in range(2):
        r = _strip_think(_qwen.call_openrouter(p, max_tokens=8000))
        if r.strip():
            log["translated_prompt"] = r
            return r
    log["translated_prompt"] = ""
    log["translate_failed"] = True
    return full                            # genuine fallback to English


def anchored_value(row, log):
    """Run 6: two-model language ID -> anchored prompt (in-language if confident) ->
    escalated SC. Returns a float or None."""
    subject = row["SubjectEntity"]
    lang, conf = identify_language(subject, log)
    eng = ANCHORED.format(subject=subject)
    if conf >= LANG_THRESH and lang:
        prompt = translate_prompt(eng, lang, log)
        log["used"] = (f"{lang} ({conf:.2f})" if log.get("translated_prompt", "").strip()
                       else f"ENGLISH (translate-failed; {lang} {conf:.2f})")
    else:
        prompt = eng
        log["used"] = f"ENGLISH (lang_conf {conf:.2f})"

    def one():
        _qwen.temperature = 0.7           # sampled answers, same as runs 1-5
        raw = _qwen.call_openrouter(prompt, max_tokens=ANSWER_MAX_TOKENS)
        return _num_from(_qwen.parse_response(raw, "hasArea"))
    return escalated_sc(one)


# --- one run over all entities, written to its own run file ----------------
def do_run(run_name, value_fn, rows, out_path, done, rawf):
    """Compute `value_fn` for every entity not yet done in this run, writing each
    entity's SC value to out_path (seeded from the hasArea rows so assemble can read it)."""
    if os.path.exists(out_path):
        existing = read_jsonl_file(out_path)
    else:
        existing = [dict(r, ObjectEntities=[]) for r in rows]   # seed one row per entity
    idx = {r["SubjectEntity"]: i for i, r in enumerate(existing)}

    todo = [r for r in rows if (run_name, r["SubjectEntity"]) not in done]
    print(f"[{run_name}] {len(rows)} entities ({len(rows)-len(todo)} done, {len(todo)} to do)",
          flush=True)
    for j, row in enumerate(todo):
        subject = row["SubjectEntity"]
        log = {"run": run_name, "subject": subject}
        v = value_fn(row, log)
        preds = [format_number(v)] if v is not None else []
        log["pred"] = preds
        if subject in idx:
            existing[idx[subject]]["ObjectEntities"] = preds
        rawf.write(json.dumps(log, ensure_ascii=False) + "\n"); rawf.flush()
        with open(out_path, "w") as f:
            for r in existing:
                f.write(json.dumps(r) + "\n")
        print(f"  [{run_name}] {j+1}/{len(todo)} {subject[:24]:<24} [{log['used']}] -> {preds}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description="Generate all six hasArea runs (single file)")
    ap.add_argument("--input", required=True, help="entities file, e.g. data/test.jsonl")
    ap.add_argument("--output-dir", default="data/hasArea_runs",
                    help="directory for the six run files")
    ap.add_argument("--rawlog", default="logs/raw_hasArea.jsonl")
    a = ap.parse_args()

    rows = [r for r in read_jsonl_file(a.input) if r["Relation"] == "hasArea"]
    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.rawlog) or ".", exist_ok=True)

    done = set()
    if os.path.exists(a.rawlog):
        for l in open(a.rawlog, encoding="utf-8"):
            try:
                d = json.loads(l); done.add((d["run"], d["subject"]))
            except Exception:
                pass
    rawf = open(a.rawlog, "a", encoding="utf-8")

    # Runs 1-5: base prompt.  Run 6: anchored native-language.
    runs = [(f"run{i}", base_value) for i in range(1, 6)] + [("run6_anchored", anchored_value)]
    for run_name, value_fn in runs:
        do_run(run_name, value_fn, rows, os.path.join(a.output_dir, f"{run_name}.jsonl"),
               done, rawf)
    rawf.close()
    print(f"All 6 hasArea runs written to {a.output_dir}/. "
          f"Now run scripts/assemble_submission.py.", flush=True)


if __name__ == "__main__":
    main()
