# Token Usage and Cost Report

**Run:** final full-dataset run of `code/main.py` that produced the submitted `output.csv`
**Requests processed:** 250 (100% of `dataset/requests.csv`)

## Model providers and models used

**None.** This solution makes **zero LLM / vision-model API calls** at prediction time.

This was a deliberate architecture decision for this submission: no LLM API key
(Anthropic or OpenAI) was available in the development environment, and rather than
leaving message/image interpretation unimplemented, both were built as fully
deterministic, local, rule-based modules instead:

- **`code/src/messages.py`** -- extracts structured facts (confirmed/amended salary
  amounts and dates, stream-ended employment, confirmed one-off invoice income, etc.)
  from `messages.csv` using regex templates covering the two languages present in the
  dataset (English and Indonesian). No network calls.
- **`code/src/images.py`** -- extracts the missing amount for blank-`amount` events
  linked via `images.csv` using **local Tesseract OCR** (`pytesseract`, already
  installed in this environment) plus a deterministic spelled-out-amount parser
  ("Seven Hundred Four Rupees and Five Paise Only" -> 704.05) as a cross-check against
  OCR table/column noise. No network calls, no API key required.

All financial arithmetic, forecasting, plan generation, and the final affordability
decision are produced by deterministic Python (see `code/src/simulator.py`,
`code/src/planner.py`, `code/src/verifier.py`) -- never by a language model, per the
project's hybrid-architecture requirement that the LLM (if used) must never make the
final financial decision. The explanation text in `decision_explanation` is also
template-generated from the already-finalized, verified decision facts
(`code/src/explain.py`), not LLM-written.

## Model calls

| Metric | Value |
|---|---|
| LLM API calls | 0 |
| Vision/LLM image calls | 0 |
| Local OCR calls (Tesseract, no API/network) | 16 (one per image in `dataset/media/images/`, cached per `image_id` so no image is ever processed twice) |

## Tokens

| Metric | Value |
|---|---|
| Total input tokens | 0 |
| Total output tokens | 0 |
| Total tokens | 0 |
| Average tokens / request | 0 |

## Estimated cost

| Metric | Value |
|---|---|
| Estimated total cost | **$0.00** |
| Estimated cost / request | **$0.00** |

## Compute cost (informational, not LLM-related)

The only non-trivial local compute is Tesseract OCR over 16 PNG images (run once,
cached) and the deterministic 90-day cash-flow simulation for each of the 250
requests. The full run (`python code/main.py`) completes in ~25 seconds on a
standard laptop CPU, with no GPU or paid API usage of any kind.

## If an LLM were added later

The codebase is structured so a real model could be swapped in without touching the
decision engine: `code/src/messages.py` and `code/src/images.py` are the only two
extraction points, and `code/src/decision_engine` (`planner.py`/`verifier.py`) never
inspects raw text -- it only consumes the structured `MessageFact` / resolved-amount
outputs those modules already produce. Token/cost accounting would be added in those
two modules only.
