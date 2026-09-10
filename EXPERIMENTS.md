# Experiments Log — What Worked and What Didn't

This document summarizes every inference-time technique we tried per relation, with
validation macro-F1 and a verdict. The final system uses only the ✓ techniques.

**Model:** `qwen/qwen3.6-27b` (27B) via OpenRouter, closed-book, no fine-tuning.
**Metric:** macro-F1 on the validation split (numeric relations use 5% relative tolerance).
**Baseline (organizer reference system):** overall 0.313 (validation).

For the full phase-by-phase narrative, see `DOCUMENTATION.md`.

**Terminology:** the `meta_*` prompt variants (`meta_precision`, `meta_verify`,
`meta_antidefault`) were produced by **error-driven prompt refinement** — feeding the
model its own errors and asking it to propose a better prompt. This is a form of
*automatic prompt optimization* (cf. Pryzant et al., APO); we avoid the term
"meta-prompting." The `meta_*` code labels are kept as-is. The script that runs this
is `scripts/meta_prompt.py` (see README.md, "Per-relation techniques"); its original
output for these two relations is archived at `meta_prompts/`.

---

## Final system — per relation

| Relation | Final technique | Val F1 |
|---|---|---|
| countryLandBordersCountry | Single exhaustive prompt, single-shot | 0.989 |
| companyTradesAtStockExchange | Vote≥3 of 5 prompts (+anti_confusion) | 0.675 |
| personHasCityOfDeath | Vote≥4 of 4 prompts | 0.500 |
| hasArea | 6-run cluster-median ensemble (5 base SC + 1 anchored native-lang SC with qwen+gemma-3-4b two-model language ID) | 0.580 |
| hasCapacity | `country_tier` prompt + confidence-escalated SC (5→9) | 0.220 |
| awardWonBy | Union of alphabetical + year_sweep passes, each per-name vote (SC×5, ≥2/5) | 0.180 |
| **Overall** | | **0.558** |

Note: hasArea val = 0.580 uses the *exact* submitted ensemble6 (r1–r5 SC + anchored native-lang SC, sequential cluster-median). The 5-run ensemble5 (no anchored) scores 0.590 on val — the anchored 6th run slightly lowers val but was included in the test submission for diversity. We report 0.580 to match the submitted system exactly.

**Test overall (Codabench): 0.5699.**

---

## hasArea

| Technique | Val F1 | Verdict | Why |
|---|---|---|---|
| Single-shot base prompt | ~0.55 | baseline | — |
| Confidence-escalated SC (5→9), cluster-median | 0.580 | ✓ | median of densest cluster removes sampling variance |
| 5-run ensemble (r1–r5, cluster-median) | 0.590 | ✓ | averaging independent SC runs reduces variance further |
| 6-run ensemble (+ anchored native-lang, qwen+gemma two-model language ID) | 0.580 | ✓ **adopted (submitted)** | anchored run adds diversity as 1 of 6 votes; slightly lowers val (0.590→0.580) but included in the test submission |
| native_lang (simple) | 0.540 | ✗ | entity boundary shift: local-language query maps to a *different* geographic extent (e.g. sub-national territory → whole country) |
| native_lang_anchored (standalone) | ~0.54 | ✗ standalone / ✓ as ensemble member | pinning entity identity in English first partially fixes the shift, but not enough to beat plain SC alone |
| entity_aware prompt + SC | 0.560 | ✗ | over-thinks entity type; sometimes misclassifies country vs island |
| 2-step local-language (`run_2step_fair.py`) | 0.52 | ✗ | translation adds no signal — a place's area is a stored number independent of query language |

**Key finding:** multilingual querying does NOT help numeric recall. The *number* is
language-independent; what changes across languages is *which entity* the model resolves
to — an entity-boundary shift, not a knowledge gain.

---

## hasCapacity

| Technique | Val F1 | Verdict | Why |
|---|---|---|---|
| `country_tier` prompt, single-shot | 0.210 | baseline | prompt supplies regional/competition-tier capacity norms |
| `country_tier` + confidence-escalated SC (5→9) | 0.220 | ✓ **adopted** | +0.010; small variance reduction on partially-known venues |
| native_lang (simple) | 0.180 | ✗ | entity boundary shift + knowledge gap |
| native_lang_anchored ensemble (2-run) | worse | ✗ | cluster-median tie-breaking bug lets the wrong value dominate with only 2 runs |
| `tier_range` prompt | 0.210 | ✗ | explicit ranges give no benefit over country_tier |

**Key finding:** errors here are genuine knowledge gaps (obscure venues), not sampling
variance — so resampling recovers little. This is the hardest relation.

---

## personHasCityOfDeath

| Technique | Val F1 | Verdict | Why |
|---|---|---|---|
| Single prompt | ~0.43 | baseline | — |
| Vote≥2 of 3 (recent_aware + meta_antidefault + city_precision) | 0.500 | ✓ | abstain when variants disagree → high precision |
| Vote≥3 of 4 (+ recent_precise) | 0.500 | = | matches 3-way; no val gain |
| Vote≥4 of 4 | 0.500 (test P=0.870) | ✓ **adopted** | strictest agreement → very high precision on test |
| Vote≥2 of 4 | 0.440 | ✗ | too permissive, false positives |
| `committed` prompt (age heuristic) + vote-5 | 0.490 | ✗ | over-predicts cities for still-living elderly people |
| `timeline` prompt | 0.310 | ✗ | over-commits to career city |
| `timeline_native` prompt | 0.370 | ✗ | same over-commit issue |
| gemma-3-4b recent-death generator (oracle fill) | 0.360 | ✗ | recovers deaths only in an oracle setting; can't be targeted |

**Key finding:** the failure is obscurity + geographic defaulting (career city / capital) +
wrong knowledge — NOT recency. Verified missed deaths were old (2009, 2019).

---

## countryLandBordersCountry

| Technique | Val F1 | Verdict | Why |
|---|---|---|---|
| Single exhaustive prompt (directional sweep) | 0.989 | ✓ **adopted** | near-ceiling; subjects are well-known countries |
| Majority voting across variants | ~same | ✗ | errors are systematic across prompts — same neighbors missed, so voting adds nothing |

---

## companyTradesAtStockExchange

| Technique | Val F1 | Verdict | Why |
|---|---|---|---|
| `simple` prompt alone | ~0.60 | baseline | — |
| Vote≥2 of 4 (simple + listed_check + meta_precision + meta_verify) | 0.672 | superseded | precision from cross-prompt agreement (intermediate config) |
| Vote≥3 of 5 (+ anti_confusion) | 0.675 | ✓ **adopted (submitted)** | anti_confusion fixes NYSE↔Nasdaq confusions; higher precision |
| anti_confusion alone, single-shot | 0.655 | ✗ standalone | good idea, needs the ensemble |
| 5B verifier (Llama-3.2-3B) | 0.417 | ✗ | verification isn't easier than generation below the knowledge boundary |

**Key finding:** never list exchange names in the prompt — it primes the model to
pattern-match named exchanges instead of recalling the company's actual listing.
US errors = NYSE↔Nasdaq confusion; others = wrong-country default.

---

## awardWonBy

| Technique | Val F1 | Verdict | Why |
|---|---|---|---|
| Alphabetical sweep + per-name confidence vote (SC×5, ≥2/5) | 0.180 | strong single pass | sweeping by letter avoids year-by-year reasoning loops; per-name vote filters hallucinations |
| year_sweep standalone | 0.133 | ✗ standalone | on its own the model stalls in loops over individual years |
| **alphabetical + year_sweep union** | 0.180 (+0.028 self-gold) | ✓ **adopted (submitted)** | union recovers names the alphabetical pass misses on huge sets (e.g. Nobel Prize in Literature, 72→128 names) |

**Key finding:** this relation is enumeration/recall-bound, NOT obscurity-bound. Missing
recipients are as notable as recalled ones; gold sets reach 610 names. Sampling can't
fix a truncation limit.

---

## Cross-cutting findings (the paper's spine)

1. **Notability predicts accuracy.** Per-entity accuracy correlates with Wikipedia
   language-edition count (pooled Pearson r ≈ 0.6). In every relation, correct entities
   are more notable than wrong ones.

2. **Selective prediction works.** Ranking entities by notability and answering only the
   most-notable fraction raises mean F1 from 0.48 (full) → 0.72 (40% coverage) → 0.88 (20%).

3. **Multilingual entity-boundary shift.** Querying in an entity's local language changes
   which geographic entity the model resolves to, not the precision of the number.
   Anchoring identity in English first partially mitigates this.

4. **Per-relation error mechanisms differ:** companyTrades = obscurity; awardWonBy =
   enumeration limit; personDeath = obscurity + geographic defaulting + wrong knowledge;
   hasCapacity = genuine knowledge gaps on obscure venues.
