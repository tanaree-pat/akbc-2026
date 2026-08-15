# LM-KBC 2026 — Shared Task System

System submission for the **AKBC/LM-KBC 2026 Shared Task** (EMNLP 2026).

Given a `(SubjectEntity, Relation)` pair, predict the complete set of correct object strings using a language model. Six relations, scored by macro-F1 (relations weighted equally).

## Constraints

- **Closed-book** — no web search, RAG, or external KB lookup at inference time.
- **≤ 32 B parameters** across all inference-time models. This system uses `qwen/qwen3.6-27b` (27 B) as the sole inference model.
- **No fine-tuning.** Post-processing (normalization, dedup, clustering, voting) is allowed.

## Results

### Validation (val.jsonl, 478 rows)

| Relation | Val F1 |
|---|---|
| countryLandBordersCountry | 0.989 |
| companyTradesAtStockExchange | 0.675 |
| personHasCityOfDeath | 0.500 |
| hasArea | 0.580 |
| hasCapacity | 0.220 |
| awardWonBy | 0.180 |
| **Overall (macro)** | **0.558** |

### Test (Codabench, principled v3 submission)

Overall macro-F1: **0.5699**

## Per-relation techniques

| Relation | Technique |
|---|---|
| hasArea | 6-run cluster-median ensemble: 5 runs with base prompt (SC 5→9, conf≥0.80) + 1 run with `native_lang_anchored` prompt (SC 5→9, conf≥0.80) |
| hasCapacity | `country_tier` prompt + confidence-escalated SC (5→9, conf≥0.80) |
| personHasCityOfDeath | Vote≥4 of 4 prompt variants: `recent_aware`, `meta_antidefault`, `city_precision`, `recent_precise` |
| countryLandBordersCountry | Single exhaustive prompt, single-shot |
| companyTradesAtStockExchange | Vote≥3 of 5 prompt variants: base (`simple`), `listed_check`, `meta_precision`, `meta_verify`, `anti_confusion` |
| awardWonBy | Alphabetical sweep (`alphabetical` prompt) + per-name confidence vote (SC×5, ≥2/5 samples agree) |

## Setup

**Python environment:** The inference scripts require the `openai` package. We recommend using a conda environment:

```bash
conda create -n lm-kbc-2026 python=3.11
conda activate lm-kbc-2026
pip install -r requirements.txt
```

**API key:** This system uses [OpenRouter](https://openrouter.ai/) to access `qwen/qwen3.6-27b`. Only the `openrouter` provider is included in this repository. Set your API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Never hardcode the API key in any script.

## Reproducing predictions

Run each relation's script, then assemble into a single submission file.

```bash
# Step 1: Run each relation
bash scripts/run_hasArea.sh data/test.jsonl          # 6 runs (~600 API calls)
bash scripts/run_hasCapacity.sh data/test.jsonl data/hasCapacity_out.jsonl
bash scripts/run_personDeath.sh data/test.jsonl      # 4 variants
bash scripts/run_countryBorders.sh data/test.jsonl data/countryBorders_out.jsonl
bash scripts/run_companyTrades.sh data/test.jsonl    # 5 variants
bash scripts/run_awardWonBy.sh data/test.jsonl data/awardWonBy_out.jsonl

# Step 2: Assemble into final submission
python scripts/assemble_submission.py \
    --input data/test.jsonl \
    --output data/predictions_v6.jsonl
```

All scripts seed their output files from the input file and overwrite predictions row-by-row (crash-safe). Pass `data/val.jsonl` as the first argument to run on the validation set instead.

### Inference modes (run_relation.py flags)

| Flags | Mode |
|---|---|
| (none) | Single-shot |
| `--samples N --samples-escalate M --confidence-min c` | Confidence-escalated SC: draw N samples; if cluster agreement < c, draw up to M total |
| `--award-samples N --confidence-min c` | awardWonBy per-name confidence vote: keep names appearing in ≥ c fraction of samples |
| `--prompt-style NAME` | Select a prompt variant from `models/prompts.py` (PROMPT_VARIANTS) |

## Evaluating predictions

```bash
python evaluate.py -p data/predictions_v6.jsonl -g data/val.jsonl
```

The provided `data/predictions_v6.jsonl` contains our final test-set predictions. Evaluating these against `val.jsonl` will yield garbage scores (different entities) — use the val predictions file instead:

```bash
python evaluate.py -p data/val_predictions_v6.jsonl -g data/val.jsonl
```

Expected overall macro-F1: **0.558**

## Repository structure

```
run_relation.py              Main inference script
evaluate.py                  Scoring script (macro-F1)
requirements.txt             Python dependencies
models/
  abstract_model.py          Base class for model backends
  openrouter_model.py        OpenRouter API client (primary)
  prompts.py                 All prompt templates and variants
data/
  train.jsonl                Training examples (from organizers)
  val.jsonl                  Validation set (from organizers)
  val_predictions_v6.jsonl   Our validation predictions (v6 system)
  predictions_v6.jsonl       Our test predictions (Codabench submission)
scripts/
  run_hasArea.sh             Reproduce hasArea (6-run ensemble)
  run_hasCapacity.sh         Reproduce hasCapacity (SC)
  run_personDeath.sh         Reproduce personDeath (4-way vote)
  run_countryBorders.sh      Reproduce countryBorders
  run_companyTrades.sh       Reproduce companyTrades (5-way vote)
  run_awardWonBy.sh          Reproduce awardWonBy
  assemble_submission.py     Assemble per-relation outputs into final file
```

## Key findings

- **Entity notability predicts accuracy.** Wikipedia language-edition count (a proxy for notability) correlates with per-entity F1 (pooled Pearson r ≈ 0.6).
- **Selective prediction.** Sorting entities by notability and predicting only the top fraction raises mean F1 from 0.48 (full set) to 0.72 (40% coverage) to 0.88 (20% coverage).
- **Multilingual entity boundary shift.** Querying in the entity's local language causes the model to map names to different geographic extents (e.g., sub-national territory → whole country in some languages). An anchored variant that explicitly pins the entity identity before switching language mitigates this.
