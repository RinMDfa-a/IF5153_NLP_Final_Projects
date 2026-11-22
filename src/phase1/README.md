# Phase 1: review relevance filtering

Decides whether a Steam review talks about **its own game** (review = premise, game context = hypothesis) and returns a
score in [0, 1]. Model: `roberta-base` cross-encoder fine-tuned on review/game pairs (test F1 0.79, precision 0.84,
recall 0.74 on 600 pairs with auto labels; see `models/phase1/task_adapted/metrics.json`).
BEAM diagram: [docs/phase1_beam.drawio](../../docs/phase1_beam.drawio).

Calling Phase 1 through ngrok (Phase 3 CLI → Phase 1 API), where the formats are written down:
- **Expected input:** the request body of `POST /phase1/filter` (keyed `{"1": {"app_id", "reviews": [...]}}`), section
  "POST /phase1/filter" in [docs/phase1_api.md](../../docs/phase1_api.md).
- **Expected output:** the response of the same endpoint (`app_name`, `reviews`, `relevance_scores`, `scored`), same section of
  [docs/phase1_api.md](../../docs/phase1_api.md). Error codes and a Python client example are further down in that file.
- **Batch files instead of the API** (`phase2_input.json`, `relevant_reviews.csv`): see
  [Outputs for Phase 2 / Phase 3 / annotators](#outputs-for-phase-2--phase-3--annotators) below.

Everything runs from the **repository root**.

## Setup

1. Python 3.11 with CUDA torch. Either the existing `venv_nlp_week7` + `pip install fastapi uvicorn`, or `uv sync`
   (add `--extra train` to retrain).
2. Model (not in git, ~480 MB): put the folder from Drive at `models/phase1/task_adapted/` (`model_info.json` + `model/`).
3. Data (not in git, only needed for batch scoring / retraining): `data/raw/`, `data/interim/`, `data/processed/` from
   Drive, or rebuild them with the pipeline below (raw sources: see CLAUDE.md).
   The API only needs `data/phase1/game_context.parquet`, which is tracked in git.

## Outputs for Phase 2 / Phase 3 / annotators

```
python -m src.phase1.inference.score_reviews  --config configs/phase1.yaml   # GPU (RTX 4060): ~3 min per 100k reviews, ~35 min for the default 500 per game
python -m src.phase1.inference.export_outputs --config configs/phase1.yaml   # seconds
```

`data/output/`:

| File | What | For |
|---|---|---|
| `relevant_reviews.parquet` | every scored review: `review_id, app_id, app_name, review_text, relevance_score, is_relevant, score_band, rejected_by, model_run` | analysis, `src/test/load_phase1_output.ipynb` |
| `relevant_reviews.csv` | relevant reviews only: `review_id, app_id, app_name, review_text, relevance_score, score_band` | **Phase 2**: set `INPUT_FILE` in the Phase 2 `main.py` to this file, no other change needed (checked with `ABSAPipeline`) |
| `annotation_check.txt` | **After the model**: 100 real reviews per score band + 100 quality-rule rejects, each with the score the trained model gave, the game context, and empty `annotator_label`, `notes` columns. `\|`-separated (see below) | **annotators** (check the model) |
| `phase2_input.json` | `{"1": {"app_id", "app_name", "reviews": [...], "relevance_scores": [...]}}`, top 30 per game | JSON in the Phase 2/3 shape |

Sizes are set in `configs/phase1.yaml` (`inference.max_reviews_per_game`, `export.*`).

### Annotation files (two of them, `|`-separated `.txt`)

| File | When | Rows | Has a score? |
|---|---|---|---|
| `annotation/to_label.txt` (`data.export_annotation`, step 7) | **before training**, builds the gold test set | 600 test-split pairs (positive / hard_negative / random_negative) with `auto_label`, empty `gold_label` | no, no model exists yet |
| `data/output/annotation_check.txt` (`inference.export_outputs`) | **after training**, checks the model | real reviews paired with their own game | yes |

Both are written by `utils.io.write_annotation_file`: one line per row, columns separated by `|` (a `|` inside the text
becomes `¦`, line breaks become spaces), no quotes, UTF-8 with BOM. They are `.txt` and not `.csv` because Excel splits a
`.csv` on the Windows list separator (`;` on en-ID) as soon as it opens it, and game descriptions contain `;`.

Open in Excel: File > Open > the `.txt` (or open it with everything in column A, then Data > Text to Columns):
Delimited → Other: `|`, text qualifier `{none}` → Advanced: decimal separator `.` (otherwise en-ID reads `0.4432` wrong).
For `to_label.txt`, fill `gold_label` and save as `annotation/labeled.txt`, or Save As CSV `annotation/labeled.csv`.
`import_annotation` reads either one and detects `|`, `;`, `,` or tab.

### Score bands (threshold 0.40, margin 0.15)

| `score_band` | score | typical content |
|---|---|---|
| `off_topic` | < 0.25 | memes and junk (quality rules, score exactly 0.0 with `rejected_by` set), text about something else |
| `borderline` | 0.25 - 0.55 | generic reviews ("Nice game, would recommend") that fit almost any game. Annotators should look here first |
| `relevant` | ≥ 0.55 | reviews about this game's content |

Checked on hand-written probes (`tests/phase1/test_probe.py`): junk = 0, game-specific reviews 0.94-0.998, generic reviews
0.12-0.64 (around the threshold). Known weak spots:
- Off-topic text that talks about Steam itself ("writing this review to get the badge…") can score high (0.65).
- Recall is 0.74, so some relevant reviews end up in `off_topic` (e.g. short reviews of games with a thin store
  description). These are what `annotation_check.txt` is for. If the annotators find many of them, try the graded
  label scheme (`pairs.label_scheme: graded` in the config, then retrain).

## API (for the Phase 3 CLI, via ngrok)

```
uvicorn src.phase1.api.server:app --port 8000
ngrok http 8000          # install from ngrok.com, then `ngrok config add-authtoken <token>` once
```

Endpoints `GET /health`, `POST /phase1/score`, `POST /phase1/filter`; request/response examples in
[docs/phase1_api.md](../../docs/phase1_api.md). Set `PHASE1_API_KEY` before starting the server to require an
`X-API-Key` header (recommended, since the ngrok URL is public).

## Score your own file (no server)

```
python -m src.phase1.inference.predict --input reviews.csv --game-context data/phase1/game_context.parquet --output scored.csv
```

The input needs `review_text` plus `app_id`, or `context_text`, or `genres` + `description`.

## Training pipeline (rebuild everything)

Main entry point: `notebooks/phase1_relevance.ipynb`. Same steps from the CLI (set `limit` in the config for a small run):

```
python -m src.phase1.data.load_reviews      --config configs/phase1.yaml
python -m src.phase1.data.quality_filter    --config configs/phase1.yaml
python -m src.phase1.data.build_context     --config configs/phase1.yaml
python -m src.phase1.data.sample_reviews    --config configs/phase1.yaml
python -m src.phase1.data.split             --config configs/phase1.yaml
python -m src.phase1.data.make_pairs        --config configs/phase1.yaml
python -m src.phase1.data.export_annotation --config configs/phase1.yaml
python -m src.phase1.data.import_annotation --config configs/phase1.yaml
python -m src.phase1.models.lexical_baseline    --config configs/phase1.yaml
python -m src.phase1.models.zero_shot_nli       --config configs/phase1.yaml
python -m src.phase1.models.train_cross_encoder --config configs/phase1.yaml
```

Tests: `python -m pytest` (unit tests, API with a fake model, export files, and the probe check when the model is present).
