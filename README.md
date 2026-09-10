# AKBC Shared Task 2026

System submission for the **AKBC Shared Task 2026** (EMNLP 2026).

Given a `(SubjectEntity, Relation)` pair, predict the complete set of correct object strings using a language model. Six relations, scored by F1 macro-averaged over all `(subject, relation)` pairs.

## Paper

**DataMind at AKBC Shared Task 2026: Relation-Specific Multiple Prompting for Closed-Book Knowledge Base Completion**

PDF: [`paper/DataMind_at_AKBC_Shared_Task_2026.pdf`](paper/DataMind_at_AKBC_Shared_Task_2026.pdf)

We combine relation-specific multiple prompting, consistency-based aggregation, and
error-driven prompt refinement to reach a validation macro-F1 of 0.558 and test
macro-F1 of 0.5699 (organizer baselines: 0.313 / 0.296), within a 32B parameter
budget and with no retrieval or fine-tuning.


## Constraints

- **Closed-book** — no web search, RAG, or external KB lookup at inference time.
- **≤ 32 B parameters** across all inference-time models. The primary model is `qwen/qwen3.6-27b` (27 B); a small `google/gemma-3-4b-it` (4 B) is used only for language identification in the hasArea anchored run, for a total of 31 B.
- **No fine-tuning.** Post-processing (normalization, dedup, clustering, voting) is allowed.

## Results

### Validation (legacy snapshot, 478 rows)

| Relation | Val F1 |
|---|---|
| countryLandBordersCountry | 0.989 |
| companyTradesAtStockExchange | 0.675 |
| personHasCityOfDeath | 0.500 |
| hasArea | 0.580 |
| hasCapacity | 0.220 |
| awardWonBy | 0.180 |
| **Overall** | **0.558** |


### Test (Codabench)

| Relation | Test F1 |
|---|---|
| countryLandBordersCountry | 0.944 |
| companyTradesAtStockExchange | 0.767 |
| personHasCityOfDeath | 0.490 |
| hasArea | 0.570 |
| hasCapacity | 0.225 |
| awardWonBy | 0.275 |
| **Overall** | **0.5699** |

Organizer baseline (Qwen 3.5 9B, simple prompting): 0.296 test / 0.313 validation.

**Data version.** The organizers corrected `train/val/test.jsonl` on 2026-08-07
(entity disambiguation, a few gold fixes) — val/test went from 478/477 rows to 475
each. The numbers below are on the *pre-correction* snapshot, archived at
[`data/legacy_2026-08-07_snapshot/`](data/legacy_2026-08-07_snapshot/) so the table
is exactly reproducible (see "Evaluating predictions"). `data/predictions.jsonl` is,
byte-for-byte, the file submitted to the Codabench test leaderboard (0.5699) — these
are the official measured numbers, not a re-run on the corrected data.

## Per-relation techniques

| Relation | Technique |
|---|---|
| hasArea | 6-run cluster-median ensemble: 5 runs with base prompt (SC 5→9, conf≥0.80) + 1 anchored native-language run with **two-model language ID** — `qwen/qwen3.6-27b` proposes the entity's local language + confidence; if <0.7, `google/gemma-3-4b-it` is also asked and the higher-confidence guess wins; if final confidence ≥0.7 the `native_lang_anchored` prompt is issued in that language (entity name kept in English), else English; same SC 5→9 |
| hasCapacity | Single prompt with confidence-escalated SC (5→9) |
| personHasCityOfDeath | Ensemble using agreement from all 4 prompts, `recent_aware`, `meta_antidefault`, `city_precision`, and `recent_precise`, else abstain  |
| countryLandBordersCountry | Single exhaustive prompt, single-sample |
| companyTradesAtStockExchange | Ensemble vote, keep if ≥3 of 5 prompts agree (base (`simple`), `listed_check`, `meta_precision`, `meta_verify`, `anti_confusion`), else abstain |
| awardWonBy | Union of alphabetical + year-by-year sweeps, each with per-name confidence vote (SC×5, ≥2/5) |

**awardWonBy answer-recovery step (not shown in the table above).** For this relation
only, if the model's response never produces a parseable JSON array (it runs out of
budget mid-reasoning), `OpenRouterModel._extract_from_thinking` (`models/openrouter_model.py`)
recovers an answer in two stages before that sample is counted as empty:
1. A second call to the same model asks it to extract every recipient name from its own
   reasoning text and return them as a JSON array.
2. If that call *also* fails to produce parseable JSON, a regex fallback
   (`_regex_names_from_thinking`) mines quoted, bulleted, and numbered names from the
   tail of the reasoning text without another API call.

This only fires for `awardWonBy` — for every other relation an unparseable/empty
response is left as `[]`, since an empty answer can be a legitimate prediction there
(e.g. "not publicly traded"). Anyone reproducing `awardWonBy` from the prompts in the
paper alone, without this recovery step, should expect a somewhat lower score, since
this step is a real part of how the shipped predictions were generated.

**Error-driven prompt refinement (how the `meta_*` prompt variants were produced).**
The paper describes producing some prompt variants by having the model diagnose its
own failures and propose improved templates. That step is `scripts/meta_prompt.py`:
given a relation, a predictions file, and (optionally) a run log with raw reasoning
traces, it builds a prompt showing the model its current prompt plus real failure
cases (prediction vs. gold) and asks it to diagnose the dominant failure pattern and
write two improved templates. Candidates are written to `meta_prompts/{relation}.txt`
for human review. The `meta_precision`,
`meta_verify`, and `meta_antidefault` variants used in the final
`companyTradesAtStockExchange` and `personHasCityOfDeath` ensembles were produced this
way; the original generated candidates are archived at
[`meta_prompts/`](meta_prompts/) for reference. Example:

```bash
python scripts/meta_prompt.py --relation companyTradesAtStockExchange \
    --pred-file data/val_predictions.jsonl \
    --base-style simple \
    --log logs/companyTrades_simple.jsonl
```

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

Run each relation's script, then assemble into a single submission file. Use `data/test.jsonl`
to reproduce the submission, or `data/val.jsonl` to evaluate locally against gold.

```bash
# Step 1: Run each relation
python scripts/run_hasArea.py --input data/test.jsonl        # 6 runs -> data/hasArea_runs/
python scripts/run_hasCapacity.py --input data/test.jsonl --output data/hasCapacity_out.jsonl
python scripts/run_personDeath.py --input data/test.jsonl    # 4 variants
python scripts/run_countryBorders.py --input data/test.jsonl --output data/countryBorders_out.jsonl
python scripts/run_companyTrades.py --input data/test.jsonl  # 5 variants
python scripts/run_awardWonBy.py --input data/test.jsonl     # 2 passes -> data/awardWonBy_runs/

# Step 2: Assemble into final submission
python scripts/assemble_submission.py \
    --input data/test.jsonl \
    --output data/predictions.jsonl
```

All scripts seed their output files from the input file and overwrite predictions row-by-row (crash-safe).

### Inference modes (run_relation.py flags)

| Flags | Mode |
|---|---|
| (none) | Single-shot |
| `--samples N --samples-escalate M --confidence-min c` | Confidence-escalated SC: draw N samples; if cluster agreement < c, draw up to M total |
| `--award-samples N --confidence-min c` | awardWonBy per-name confidence vote: keep names appearing in ≥ c fraction of samples |
| `--prompt-style NAME` | Select a prompt variant from `models/prompts.py` (PROMPT_VARIANTS) |

## Evaluating predictions

```bash
python evaluate.py -p data/predictions.jsonl -g data/val.jsonl
```

`data/predictions.jsonl` is the test-set predictions file — use `data/val_predictions.jsonl` to evaluate on val:

```bash
python evaluate.py -p data/val_predictions.jsonl -g data/val.jsonl
```

That gives **0.554** — scored against the current, corrected `data/val.jsonl`. **To
exactly reproduce the 0.558 reported above**, score against the archived legacy
snapshot instead (see "Data version" under Results):

```bash
python evaluate.py -p data/val_predictions.jsonl -g data/legacy_2026-08-07_snapshot/val.jsonl
```

This reproduces every figure in the Results table exactly.

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
  train.jsonl                Training examples (from organizers, current/corrected snapshot)
  val.jsonl                  Validation set (from organizers, current/corrected snapshot)
  test.jsonl                 Test inputs, no gold (from organizers, current/corrected snapshot)
  val_predictions.jsonl      Our validation predictions (generated on the legacy snapshot below)
  predictions.jsonl          Our test predictions (Codabench submission; legacy snapshot below)
  legacy_2026-08-07_snapshot/ train.jsonl / val.jsonl / test.jsonl as they stood before the
                             organizers' 2026-08-07 correction -- the exact data the Results
                             table and the predictions above were computed on
scripts/
  run_hasArea.py             Generate all 6 hasArea runs (5 base SC + 1 two-model
                             anchored native-language); ensemble in assemble_submission.py
  run_hasCapacity.py         Reproduce hasCapacity (SC)
  run_personDeath.py         Reproduce personDeath (4-way vote)
  run_countryBorders.py      Reproduce countryBorders
  run_companyTrades.py       Reproduce companyTrades (5-way vote)
  run_awardWonBy.py          Reproduce awardWonBy (alphabetical + year_sweep union)
  assemble_submission.py     Assemble per-relation outputs into final file
  meta_prompt.py              Error-driven prompt refinement (produces meta_* prompt
                              candidates in models/prompts.py; see "Per-relation
                              techniques" above)
meta_prompts/                 Archived meta_prompt.py output for companyTradesAtStockExchange
                              and personHasCityOfDeath -- the candidates that became
                              the meta_* prompt variants
```
