# Buy or Wait? -- Solution

Deterministic, rule-based financial affordability engine for the HackerRank
Orchestrate "Buy or Wait?" challenge. No LLM API key was available in this
environment, so message and image interpretation are implemented as local,
deterministic modules (regex-based fact extraction and local Tesseract OCR)
rather than left unimplemented -- see `evaluation/usage_report.md` for the
full explanation of that decision. The final financial decision is always
produced by deterministic Python and passes a deterministic verifier before
being written out; nothing in this solution calls an external API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
```

Image amount extraction uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract),
which must be installed separately (it is a system binary, not a pip package):

- Windows: install from the UB-Mannheim build and it will be auto-detected at
  `C:\Program Files\Tesseract-OCR\tesseract.exe` if not already on `PATH`.
- macOS: `brew install tesseract`
- Linux: `apt-get install tesseract-ocr`

If Tesseract is unavailable, the pipeline still runs -- the 16 affected events
(blank `amount`, linked to an image) are simply excluded from that user's
forecast rather than guessed at (never silently treated as zero).

## Run

From the repository root:

```bash
python code/main.py
```

This reads `dataset/requests.csv` (and all supporting dataset files), runs the
full pipeline for all 250 requests, and writes the completed predictions to
`output.csv` in the repository root. Takes about 25 seconds on a laptop CPU.

## Validate

```bash
python code/evaluation/validate_output.py   # schema/bounds/plan-consistency checks on output.csv
python code/evaluation/test_units.py        # unit tests for the simulator, currency, planner
python code/evaluation/test_solver.py       # regression check against dataset/sample_requests.csv
```

## Architecture

```text
code/
├── main.py                    entry point
├── src/
│   ├── data_loader.py         loads + indexes all dataset CSVs by user/request
│   ├── currency.py            fixed-rate FX conversion (exchange_rates.csv), BFS-chained
│   ├── messages.py            deterministic English/Indonesian fact extraction from messages.csv
│   ├── images.py              local OCR amount extraction for blank-amount events
│   ├── events.py              recurrence detection + 90-day forward forecast construction
│   ├── models.py               shared dataclasses (ForecastItem, SpendingChangeOption)
│   ├── payment_options.py     parses request_payment_options.csv installment schedules
│   ├── simulator.py           daily balance trajectory, amount_safe_to_pay, earliest-safe-date
│   ├── spending_changes.py    minimal stop/reduce combination search
│   ├── planner.py             candidate plan generation, tiered status selection, ranking
│   ├── verifier.py            final safety-net validation before a row is emitted
│   ├── explain.py             template-based decision_explanation (no LLM)
│   └── pipeline.py            orchestrates one request end-to-end
└── evaluation/
    ├── validate_output.py     output.csv schema/contract validator
    ├── test_units.py          unit tests (boundaries, currency, partial payment, ...)
    ├── test_solver.py         regression check against sample_requests.csv
    └── usage_report.md        token/cost report for the final run (zero -- no LLM used)
```

## Key design decisions

- **`current_available_balance` is the balance at that user's `request_date`.**
  Each user in this dataset has exactly one request, and each user's
  `financial_events.csv` history ends right before their `request_date` --
  so historical events are used only to detect recurring patterns, never
  replayed into the starting balance (that would double-count).
- **Recurrence detection** groups a user's settled history (plus any
  `scheduled` future events, which can be the *only* evidence of a new job's
  cadence -- e.g. a prorated first salary + a next-confirmed-salary event) by
  category, filters amount outliers against the *recent* median (so a
  one-off bonus or an old employer's rate doesn't distort a new pattern),
  and detects a fixed day-of-month when present (most recurring bills and
  salaries in this dataset land on the same day every month) rather than a
  drifting average day-count interval. A "Final ... payroll"-style
  description on the latest occurrence ends that income stream.
- **`amount_safe_to_pay`** and **`earliest_date_for_full_payment`** both
  reduce to a closed-form suffix-minimum over a single 90-day daily balance
  array (paying `X` today shifts every later day's balance down by `X`
  uniformly), so both are exact given the forecast, not search/binary-search
  approximations.
- **Status/method selection** follows a two-stage process: gather every
  preference-eligible, verified-safe candidate plan (full payment,
  full-payment-with-flexible-changes, partial payment, each supplied
  installment option, wait), keep only the candidates in the
  highest-precedence tier that has any (`affordable_now` >
  `affordable_with_plan` > `affordable_later`), then apply the six ranking
  rules from the problem statement to pick a winner within that tier.
- **Flexible spending changes** always target the category's *most recent*
  real `event_id` (the natural reading of the sample data, where
  `stop:`/`reduce_to:` references are always that user's latest occurrence
  of the category) and reduce to exactly `minimum_allowed_amount` when
  reducing, matching the supplied sample outputs.
- **Deadline compliance is a hard eligibility gate, not just a ranking
  preference**, for every method except `wait`: "the plan must complete the
  request by desired_completion_date" (90-Day Safety Check) is part of what
  makes a candidate plan *safe*, so a full-payment/partial/installment
  candidate that finishes after the deadline is discarded before ranking
  begins, rather than merely ranked last. This is enforced in both
  `planner.py` and, independently, `verifier.py`.
- **An explicit forward-dated event only dedupes against a recurring
  projection when its amount is also close** (not just its date) -- two
  events sharing a category (e.g. a one-off pending "shopping" debit and an
  unrelated recurring "shopping" pattern) are genuinely different cash
  events and must both be counted. Matching on date-proximity alone silently
  dropped real, often large, pending debits.
- **A category can hold two concurrent recurring streams** (this dataset
  explicitly models "primary" + "second household income" earners under one
  `salary` category). When a category's recent amounts split into two
  well-separated, evenly-sized clusters, each cluster is fit as its own
  independent series and summed, instead of one blended (and therefore
  wrong-cadence, wrong-amount) series.
- **`max_installment_months` counts monthly *payments*, not payment span in
  days.** Every installment option in this dataset pays roughly monthly, and
  profile values are whole numbers (2-12); a 3-payment plan is "3 months",
  not `(last_payment - first_payment) / 30 ≈ 2` months.
- **A day-of-month consistency pass runs alongside the amount-outlier
  filter.** A stray point can pass the amount-outlier band (close enough in
  magnitude) yet still land on the wrong day of the month -- e.g. one
  commission payment whose amount happens to be close to a base salary --
  and that single date corrupts the average interval enough to miss the
  monthly-day detection entirely. When a large majority of points share a
  day-of-month, the minority is dropped before fitting cadence, mirroring
  the amount-outlier filter but on the date axis.

## Known limitations

- Recurrence/forecast amounts are statistical estimates (recent-window mean,
  with outlier filtering); they will not exactly reproduce a hidden
  ground-truth generator's precise formula, so `amount_safe_to_pay` should be
  read as "close" rather than bit-exact -- validated against
  `dataset/sample_requests.csv` via `test_solver.py`. A dedicated audit
  (`evaluation/estimator_sweep.py`) tested median/trimmed-mean/weighted/
  latest-observation estimators and a systematic expense-conservatism bias
  against all 25 public samples: no alternative estimator improved on a
  plain recent-window mean, and any conservatism bias strong enough to move
  the average error traded categorical accuracy away (regressions on
  `recommended_payment_method`/`payment_plan` that got worse the more bias
  was applied) rather than fixing it -- the remaining gap is consistent with
  per-request noise in an unobservable synthetic-generation formula, not a
  further fixable estimator choice.
- OCR-based image amount extraction is best-effort; it resolves the intended
  figure cleanly on most of the 16 linked images but is unreliable on
  handwritten receipts (no vision model is available in this environment).
