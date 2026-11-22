# IF5153_NLP_Final_Projects

Steam review summary generation, IF5153 NLP Group 5.

| Phase | Task | Code | Branch |
|---|---|---|---|
| 1 | Review relevance filtering (cross-encoder, NLI-style) | `src/phase1/` ([README](src/phase1/README.md), [API](docs/phase1_api.md)) | `feat/phase1` |
| 2 | Aspect extraction (ABSA, T5/BART) | `src/phase2/` | `feat/phase2` |
| 3 | Summary generation + demo CLI | `src/` | `feat/phase3` |

Phase 1 → Phase 2: `data/output/relevant_reviews.csv` (batch) or `POST /phase1/filter` (API, served via ngrok).
