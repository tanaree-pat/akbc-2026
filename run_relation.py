import argparse
import json
import statistics
import time

from evaluate import try_parse_number

NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}

# Markers that indicate the model is uncertain about its answer.
# Used by confidence-gated SC (--samples-adaptive).
_UNCERTAINTY_MARKERS = [
    "not sure", "uncertain", "approximately", "roughly",
    "i think", "i believe", "might be", "could be",
    "possibly", "probably", "unsure", "unclear",
    "i'm not", "don't know", "no data", "no information",
    "hard to say", "difficult to say", "no record",
]


def _is_uncertain(raw: str) -> bool:
    """Return True if ≥2 uncertainty markers appear in the raw model response."""
    lower = raw.lower()
    return sum(1 for m in _UNCERTAINTY_MARKERS if m in lower) >= 2


def make_model(provider: str, model_name: str, temperature: float = 0.0,
               prompt_style: str = "simple", verify: bool = False, verbose: bool = False,
               log_name: str = ""):
    if provider == "openrouter":
        from models.openrouter_model import OpenRouterModel
        return OpenRouterModel(model_name=model_name, temperature=temperature,
                               prompt_style=prompt_style, verify=verify, verbose=verbose,
                               log_name=log_name)
    elif provider == "groq":
        from models.groq_model import GroqModel
        return GroqModel(model_name=model_name, temperature=temperature,
                         prompt_style=prompt_style, verify=verify, verbose=verbose,
                         log_name=log_name)
    elif provider == "google":
        from models.google_model import GoogleAIModel
        return GoogleAIModel(model_name=model_name, temperature=temperature,
                             prompt_style=prompt_style, verify=verify, verbose=verbose,
                             log_name=log_name)
    else:
        from models.ollama_model import OllamaModel
        return OllamaModel(model_name=model_name, temperature=temperature,
                           prompt_style=prompt_style, verify=verify, verbose=verbose,
                           log_name=log_name)


def cluster_median(values, tol=0.05):
    """Self-consistency aggregation for numeric answers: find the densest cluster
    of samples that agree within `tol` (5% relative), then return that cluster's
    median. Robust to one-off WILD figures that a plain median would be dragged by."""
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


def cluster_confidence(values, tol=0.05):
    """Fraction of numeric values that fall in the consensus cluster (0.0–1.0).
    High value = samples agree = model is confident. Low value = spread = uncertain.
    Returns 0.0 if no values."""
    if not values:
        return 0.0
    best = []
    for v in values:
        scale = max(abs(v), 1e-9)
        supporters = [u for u in values if abs(u - v) <= tol * scale]
        if len(supporters) > len(best):
            best = supporters
    return len(best) / len(values)


def aggregate_numeric(sample_preds, tol=0.05):
    """sample_preds: list of prediction-lists (one per sample), e.g. [['5000'], [], ['4900']].
    Returns a single [value] prediction, or [] if no sample yielded a number."""
    nums = []
    for preds in sample_preds:
        if preds:
            v = try_parse_number(preds[0])
            if v is not None:
                nums.append(v)
    med = cluster_median(nums, tol=tol)
    if med is None:
        return []
    if med == int(med):
        return [str(int(med))]
    return [str(round(med, 6))]


def award_confidence_vote(sample_preds, min_conf=0.4):
    """Multi-label confidence voting for awardWonBy.
    Each unique name gets confidence = (# samples containing it) / n_samples.
    Per-sample deduplication prevents one sample inflating a name's count.
    Returns (kept_names, avg_confidence) — kept are those with conf >= min_conf."""
    from collections import Counter
    n = len(sample_preds)
    if n == 0:
        return [], 0.0
    name_count = Counter()
    canonical = {}
    for preds in sample_preds:
        seen = set()
        for name in preds:
            norm = name.strip().lower()
            if norm not in seen:
                name_count[norm] += 1
                seen.add(norm)
                if norm not in canonical:
                    canonical[norm] = name.strip()
    kept, confs = [], []
    for norm, count in name_count.items():
        conf = count / n
        if conf >= min_conf:
            kept.append(canonical[norm])
            confs.append(conf)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return kept, avg_conf


def aggregate_vote(sample_preds):
    """Categorical self-consistency: majority-vote the first item across samples.
    Used for text relations (e.g. personHasCityOfDeath) where the answer is a name.
    Empty-list samples vote for 'abstain'. Returns (preds, confidence) where
    confidence = winning_votes / total_samples (0.0–1.0). Ties break toward the
    first-seen non-empty city; if 'abstain' wins, returns []."""
    n = len(sample_preds)
    if n == 0:
        return [], 0.0
    # Normalise each sample's first answer to a canonical key; [] -> None (abstain)
    keys = []
    display = {}
    for preds in sample_preds:
        if preds and preds[0].strip():
            k = preds[0].strip().lower()
            keys.append(k)
            display.setdefault(k, preds[0].strip())
        else:
            keys.append(None)
    from collections import Counter
    counts = Counter(keys)
    winner, wins = counts.most_common(1)[0]
    confidence = wins / n
    if winner is None:
        return [], confidence
    return [display[winner]], confidence


RELATION = "personHasCityOfDeath"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation", default=RELATION)
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "groq", "ollama", "google"])
    parser.add_argument("--model", default="qwen/qwen-2.5-32b-instruct")
    parser.add_argument("--input", default="data/val.jsonl")
    parser.add_argument("--output", default="data/val_predictions.jsonl")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip the first N rows of this relation (resume after a partial run)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Self-consistency: draw N samples per row and aggregate (numeric relations only)")
    parser.add_argument("--samples-adaptive", type=int, default=0,
                        help="Confidence-gated SC: run 1 sample first; if uncertain, run up to N samples total "
                             "(numeric relations only). Mutually exclusive with --samples>1.")
    parser.add_argument("--samples-escalate", type=int, default=0,
                        help="Confidence-escalated numeric SC: draw --samples first; if the agreement "
                             "confidence is below --confidence-min, draw more up to this many total, then "
                             "re-aggregate. Numeric relations only.")
    parser.add_argument("--confidence-min", type=float, default=0.6,
                        help="Agreement threshold [0-1] below which --samples-escalate draws more samples, "
                             "or below which --vote-samples abstains ([]).")
    parser.add_argument("--vote-samples", type=int, default=0,
                        help="Categorical majority-vote SC for text relations: draw N samples, return the "
                             "plurality answer. Abstains ([]) when vote confidence < --confidence-min.")
    parser.add_argument("--award-samples", type=int, default=0,
                        help="awardWonBy multi-label SC: draw N samples, keep names with per-name "
                             "confidence >= --confidence-min (default 0.4 = 2/5 samples).")
    parser.add_argument("--award-escalate", type=int, default=0,
                        help="awardWonBy SC: escalate to this many samples if avg name confidence "
                             "< --confidence-min after the base batch.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Default 0.0 for single-shot, 0.7 when --samples>1")
    parser.add_argument("--prompt-style", default="simple",
                        help="Prompt variant to use. Options: simple, decompose, entity_aware, tier_range, "
                             "event_recall, comparison, country_tier, regional_recall, chain_of_thought, "
                             "recent_aware, city_precision")
    parser.add_argument("--verify", action="store_true",
                        help="Add a second self-verification pass that sanity-checks the numeric answer")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full raw model response (including think blocks) before parsing")
    parser.add_argument("--log-name", default="",
                        help="Log file name (without .jsonl). E.g. p15_test_hasArea_sc")
    args = parser.parse_args()

    # Sensible temperature default: deterministic single-shot, or 0.7 for sampling
    any_sampling = (args.samples > 1 or args.samples_adaptive > 1
                    or args.samples_escalate > 1 or args.vote_samples > 1)
    if args.temperature is None:
        args.temperature = 0.7 if any_sampling else 0.0

    model = make_model(args.provider, args.model, temperature=args.temperature,
                       prompt_style=args.prompt_style, verify=args.verify, verbose=args.verbose,
                       log_name=args.log_name)
    inputs = [json.loads(l) for l in open(args.input)]
    relation_rows = [r for r in inputs if r["Relation"] == args.relation]

    if args.start_from:
        print(f"Skipping first {args.start_from} rows (already done)")
        relation_rows = relation_rows[args.start_from:]

    use_award = args.award_samples > 1 and args.relation == "awardWonBy"
    use_escalate = args.samples_escalate > 1 and args.relation in NUMERIC_RELATIONS
    # Escalate takes precedence: it uses --samples as its base batch size, so plain
    # SC / adaptive must stand down when escalation is requested.
    use_sc = args.samples > 1 and not use_escalate and args.relation in NUMERIC_RELATIONS
    use_adaptive = args.samples_adaptive > 1 and not use_escalate and args.relation in NUMERIC_RELATIONS
    use_vote = args.vote_samples > 1 and not use_award  # categorical — any relation, but meant for text
    print(f"Provider: {args.provider}  Model: {args.model}  temp={args.temperature}", flush=True)
    if use_award:
        base = args.award_samples
        esc = f"→{args.award_escalate}" if args.award_escalate > base else ""
        print(f"Award SC: {base}{esc} samples/row, keep names with conf >= {args.confidence_min:.0%}", flush=True)
    if use_sc:
        print(f"Self-consistency: {args.samples} samples/row, agreement-cluster median", flush=True)
    if use_adaptive:
        print(f"Adaptive SC: up to {args.samples_adaptive} samples/row when uncertain", flush=True)
    if use_escalate:
        print(f"Confidence-escalated SC: {args.samples}→{args.samples_escalate} samples "
              f"when agreement < {args.confidence_min:.0%}", flush=True)
    if use_vote:
        print(f"Majority-vote SC: {args.vote_samples} samples/row, abstain when "
              f"agreement < {args.confidence_min:.0%}", flush=True)
    print(f"Running {len(relation_rows)} rows for {args.relation}", flush=True)

    # Load existing predictions into a mutable list, indexed for fast lookup
    existing = [json.loads(l) for l in open(args.output)]
    pred_index = {(r["SubjectEntity"], r["Relation"]): i for i, r in enumerate(existing)}

    for i, row in enumerate(relation_rows):
        if use_award:
            sample_preds = [model.generate_predictions([row])[0] for _ in range(args.award_samples)]
            preds, avg_conf = award_confidence_vote(sample_preds, min_conf=args.confidence_min)
            escalated = False
            if avg_conf < args.confidence_min and args.award_escalate > args.award_samples:
                extra = args.award_escalate - args.award_samples
                sample_preds += [model.generate_predictions([row])[0] for _ in range(extra)]
                preds, avg_conf = award_confidence_vote(sample_preds, min_conf=args.confidence_min)
                escalated = True
            tag = f"escalated→{len(sample_preds)}" if escalated else f"{args.award_samples} samples"
            print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {len(preds)} names  "
                  f"avg_conf={avg_conf:.0%} [{tag}]", flush=True)
        elif use_sc:
            sample_preds = []
            for _ in range(args.samples):
                sample_preds.append(model.generate_predictions([row])[0])
            preds = aggregate_numeric(sample_preds)
            raw = [p[0] if p else "-" for p in sample_preds]
            nums = [try_parse_number(p[0]) for p in sample_preds if p and try_parse_number(p[0]) is not None]
            conf = cluster_confidence(nums)
            print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {preds}  "
                  f"samples={raw}  conf={conf:.0%}", flush=True)
        elif use_adaptive:
            first = model.generate_predictions([row])[0]
            uncertain = _is_uncertain(getattr(model, "_last_raw", ""))
            if uncertain:
                more = [model.generate_predictions([row])[0]
                        for _ in range(args.samples_adaptive - 1)]
                sample_preds = [first] + more
                preds = aggregate_numeric(sample_preds)
                raw = [p[0] if p else "-" for p in sample_preds]
                print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {preds}  "
                      f"[uncertain, {args.samples_adaptive} samples] {raw}", flush=True)
            else:
                preds = first
                print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {preds}  [confident]", flush=True)
        elif use_escalate:
            # Draw the base batch, measure agreement; escalate only if the model disagrees with itself
            sample_preds = [model.generate_predictions([row])[0] for _ in range(args.samples)]
            nums = [try_parse_number(p[0]) for p in sample_preds if p and try_parse_number(p[0]) is not None]
            conf = cluster_confidence(nums)
            escalated = False
            if conf < args.confidence_min and args.samples_escalate > args.samples:
                extra = args.samples_escalate - args.samples
                sample_preds += [model.generate_predictions([row])[0] for _ in range(extra)]
                nums = [try_parse_number(p[0]) for p in sample_preds if p and try_parse_number(p[0]) is not None]
                conf = cluster_confidence(nums)
                escalated = True
            preds = aggregate_numeric(sample_preds)
            raw = [p[0] if p else "-" for p in sample_preds]
            tag = f"escalated→{len(sample_preds)}" if escalated else "confident"
            print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {preds}  "
                  f"samples={raw}  conf={conf:.0%} [{tag}]", flush=True)
        elif use_vote:
            sample_preds = [model.generate_predictions([row])[0] for _ in range(args.vote_samples)]
            preds, conf = aggregate_vote(sample_preds)
            if conf < args.confidence_min:
                preds = []  # not enough agreement — abstain to protect precision
            raw = [p[0] if p else "-" for p in sample_preds]
            print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {preds}  "
                  f"votes={raw}  conf={conf:.0%}", flush=True)
        else:
            preds = model.generate_predictions([row])[0]
            print(f"{i+1}/{len(relation_rows)}: {row['SubjectEntity']} → {preds}", flush=True)

        # Write to disk after every row — crash-safe, no lost work
        key = (row["SubjectEntity"], args.relation)
        if key in pred_index:
            existing[pred_index[key]]["ObjectEntities"] = preds
        with open(args.output, "w") as f:
            for r in existing:
                f.write(json.dumps(r) + "\n")

    print("Done. Predictions updated.", flush=True)


if __name__ == "__main__":
    main()
