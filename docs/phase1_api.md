# Phase 1 API (relevance filtering)

Server: `src/phase1/api/server.py` (FastAPI). Run it from the repository root on the machine that has the model:

```
uvicorn src.phase1.api.server:app --port 8000
ngrok http 8000                      # copy the https://....ngrok-free.app URL
```

In the Phase 3 CLI `.env`: `PHASE_1_BASE_ENDPOINT=https://<ngrok-url>`.
If the server was started with `PHASE1_API_KEY=<secret>`, send the header `X-API-Key: <secret>` on every `/phase1/*` call.

## Score meaning

| field | meaning |
|---|---|
| `relevance_score` | 0..1, "does this review talk about this game?" |
| `is_relevant` | `relevance_score >= threshold` (threshold 0.40, chosen on the validation set, returned by `/health`) |
| `score_band` | `off_topic` (< thr − 0.15), `borderline` (thr ± 0.15, mostly generic reviews like "Nice game"), `relevant` (≥ thr + 0.15) |
| `rejected_by` | name of the quality rule that caught junk (ASCII art, `10/10 10/10 ...`, "Early Access Review", < 3 words, non-English). Those reviews are not sent to the model and get score 0.0 |

## `GET /health`

```json
{"status": "ok", "model_run": "task_adapted", "threshold": 0.4, "device": "cuda:0", "known_games": 8747, "auth": false}
```

## `POST /phase1/filter` (pipeline: Phase 1 → Phase 2)

The body has the same keyed shape as the Phase 2/3 JSON files. Each game needs `reviews` plus one way to identify the game:
`app_id` (looked up in `data/phase1/game_context.parquet`; unknown ids are fetched from the Steam Store API), or
`context_text`, or `genres` + `description` (+ optional `categories`). Other keys (e.g. `aspects`) are passed through unchanged.

Query parameters: `top_k` (max reviews kept per game), `min_score` (default: the server's `PHASE1_MIN_SCORE` if set, else the model threshold; `0` keeps every review), `include_scored` (default true).

```
POST /phase1/filter?top_k=30
{
  "1": {"app_id": 292030, "reviews": [
        "Geralt's monster contracts are incredibly well written and the open world of Velen is stunning.",
        "It's a nice game, worth the money I think.",
        "Bought a new chair today and my back finally stopped hurting.",
        "Early Access Review"]},
  "2": {"app_id": 1086940, "reviews": ["Turn based combat with dice rolls feels like real D&D.", "My dog ate my homework lol"]}
}
```

Response: `reviews` = the kept reviews (highest score first, cut by the score cut above and by `top_k`) and can go straight to Phase 2;
`relevance_scores` is parallel to `reviews`; `scored` = every input review with its score and `is_relevant`.

```json
{
  "1": {
    "app_id": 292030,
    "app_name": "The Witcher 3: Wild Hunt",
    "reviews": ["Geralt's monster contracts are incredibly well written and the open world of Velen is stunning."],
    "relevance_scores": [0.998],
    "scored": [
      {"review_text": "Geralt's monster contracts ...", "relevance_score": 0.998, "is_relevant": true, "score_band": "relevant", "rejected_by": null},
      {"review_text": "It's a nice game, worth the money I think.", "relevance_score": 0.1215, "is_relevant": false, "score_band": "off_topic", "rejected_by": null},
      {"review_text": "Bought a new chair today ...", "relevance_score": 0.1089, "is_relevant": false, "score_band": "off_topic", "rejected_by": null},
      {"review_text": "Early Access Review", "relevance_score": 0.0, "is_relevant": false, "score_band": "off_topic", "rejected_by": "empty_or_placeholder"}
    ]
  },
  "2": {"app_id": 1086940, "app_name": "Baldur's Gate 3", "reviews": ["Turn based combat ..."], "relevance_scores": [0.9697], "scored": ["..."]}
}
```

## `POST /phase1/score` (demo: score single reviews)

```
POST /phase1/score
{"reviews": [
  {"review_text": "Digging to the underworld and beating the Wall of Flesh is great.", "app_id": 105600},
  {"review_text": "shoot zombies all night", "genres": "Action; FPS", "description": "Survive zombie waves with friends."},
  {"review_text": "lol", "context_text": "Genres: Racing. Description: cars"}
]}
```

```json
{"model_run": "task_adapted", "threshold": 0.4, "results": [
  {"review_text": "Digging to the underworld ...", "app_id": 105600, "relevance_score": 0.9288, "is_relevant": true, "score_band": "relevant", "rejected_by": null},
  {"review_text": "shoot zombies all night", "app_id": null, "relevance_score": 0.9494, "is_relevant": true, "score_band": "relevant", "rejected_by": null},
  {"review_text": "lol", "app_id": null, "relevance_score": 0.0, "is_relevant": false, "score_band": "off_topic", "rejected_by": "too_short"}
]}
```

## Errors

| status | when |
|---|---|
| 401 | `PHASE1_API_KEY` is set and the `X-API-Key` header is missing or wrong |
| 413 | more than 5000 reviews in one request (split the request) |
| 422 | a game cannot be identified (unknown `app_id` with no Steam data, or no `app_id`/context given); `detail` says which one |

## Python client example (Phase 3 CLI)

```python
import requests
from conf import settings

resp = requests.post(f"{settings.PHASE_1_BASE_ENDPOINT}/phase1/filter", params={"top_k": 30},
                     json={"1": {"app_id": 105600, "reviews": reviews}}, timeout=120)
resp.raise_for_status()
phase2_input = {k: {"app_name": g["app_name"], "reviews": g["reviews"]} for k, g in resp.json().items()}
```
