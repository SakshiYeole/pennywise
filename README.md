# Pennywise

A terminal agent for messy bank CSVs. Give it a statement export and it categorizes every
row, flags what looks wrong, and writes a report that reconciles to the file:
`counted + excluded + quarantined = rows read`.

Built for a freelancer who exports a bank CSV every month and wants totals they can trust
in under a minute, with risky rows called out instead of silently counted.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. settings: fill in LLM_API_KEY and LLM_BASE_URL in .env (see "Settings (.env)" below)
python -m pennywise.llm                      # smoke test: one tiny call, prints model + latency

# 2. run it (the demo data is already in the repo; see "Data files" to rebuild or replace it)
pennywise samples/sample.csv                 # live LLM
pennywise samples/sample.csv --mock-llm      # fully offline
streamlit run app.py                         # the same run in a web page, for recording
```

Reports land in `reports/` as `<name>-rows.csv` and `<name>-summary.csv`.

## What a run does

```
read_csv -> plan -> categorize -> (ask) -> flag_anomalies -> compute_totals -> write_report
```

| Step | Who does it | Notes |
| --- | --- | --- |
| `read_csv` | code | Encoding fallback, delimiter detection, bad rows quarantined with line number and reason |
| `plan` | LLM | One call that sees file statistics only, never merchant text. Its reply is validated against a tool allowlist; required steps always run, in order |
| `categorize` | rules, then LLM | Saved merchant rules first; the LLM only labels unknown merchants, in batches of 20, and may only answer with one of 17 categories |
| ask | you | Only for merchants the LLM was unsure about (confidence < 0.6), at most 5 per run. Answers become saved rules |
| `flag_anomalies` | code | `NEEDS_REVIEW` (zero, negative, ≥ $1,000,000 or > 20x median, excluded from totals), `POSSIBLE_DUPLICATE`, `SUSPICIOUS_TEXT` |
| `compute_totals` | code | `Decimal` only, with an independent recomputation as the reconciliation check |
| `write_report` | code | Two CSVs; text cells that start with `= + - @` are prefixed with `'` so a spreadsheet never runs them |

Every step prints a visible line (`→ categorize(rows=12)` then `✓ …`). The terminal and the
web page render the same event stream.

## Options

| Flag | Meaning |
| --- | --- |
| `--mock-llm[=ok\|fail\|inject]` | Offline. `fail` simulates an outage, `inject` simulates a hijacked model answering `$0` |
| `--confirm LINE` | Count a flagged row after you reviewed it (repeatable) |
| `--rules PATH` | Merchant rules file (default `~/.pennywise/rules.json`) |
| `--out DIR` | Folder for the CSV files (default `reports`) |
| `--ping` | One tiny LLM call; prints model and latency |

Exit codes: `0` done, `2` the file cannot be processed (empty, binary, wrong columns), `130` interrupted.

## How bad input is handled

| Input | What happens |
| --- | --- |
| Missing amount, garbage date | Row quarantined and reported; the run continues |
| Zero, negative or 10M-scale amount | Flagged, excluded from totals until confirmed |
| Prompt injection in a merchant name | Treated as data: sent as an escaped JSON string, LLM output limited to a category list, flagged `SUSPICIOUS_TEXT`, no tools reachable from the LLM |
| LLM outage, 429/5xx, expired token | Retries with 1 s / 2 s backoff (never on 401), then rules-only; unknowns stay `Uncategorized`; the run still completes |
| Empty file, binary file, wrong columns, odd encodings | Clear message and exit code 2, never a traceback |
| Runaway cost | Caps of 8 steps, 5 LLM calls and 20k tokens per run |
| Formula text in a CSV cell | Neutralized in the written report |

## Tests

```bash
pytest                 # 20 offline tests, no token needed
pytest -m live         # scores the real endpoint on 17 labeled rows (needs .env)
```

Last live scores: 16/17 and 15/17 on two runs (the model is not fully deterministic). The usual
miss is `MONTHLY INSTALLMENTS (1 OF 36)`, which the model is unsure about, so it is left Uncategorized
rather than guessed; `BELLAGIO` flipped between Entertainment and Travel once. The test passes at 15 or more.

## Settings (.env)

`.env` sits in the repo root and is **ignored by git**. It holds the endpoint settings for any
OpenAI-compatible chat API:

| Variable | Meaning |
| --- | --- |
| `LLM_API_KEY` | Access token (secret; never commit it) |
| `LLM_BASE_URL` | Base URL of the endpoint |
| `LLM_MODEL` | Model name; default `gpt-5.6-luna` |
| `LLM_AUTH_SCHEME` | Empty sends the token as-is in the `Authorization` header; `Bearer` for standard APIs |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_ATTEMPTS` | Per-call timeout (default 30) and total tries per call (default 3) |

## Data files

The demo data comes from the MIT-licensed synthetic dataset
[DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
(pinned commit `3e8d052`). It has descriptions and labels but **no dates or amounts**, so
`scripts/build_samples.py` generates those with a fixed seed. The trap rows in `sample.csv`
are hand-written. No real bank data is in this repo.

### Where each file lives

| Path | In git? | What it is | How it gets there |
| --- | --- | --- | --- |
| `data/raw/transactions-synthetic.csv` | **No** (ignored, 3.5 MB) | The raw open dataset, header `description,category`, 68,000 rows | `python scripts/fetch_data.py`, or place the file there yourself (below) |
| `data/raw/.gitkeep` | Yes | Keeps the empty folder in git | Already there |
| `data/NOTICE` | Yes | Source, commit and MIT attribution | Already there |
| `data/seed_rules.json` | Yes | 45 known-merchant rules the agent checks before calling the LLM | `scripts/build_seed.py` |
| `data/mock_answers.json` | Yes (300 KB) | Answers for the offline `--mock-llm` mode | `scripts/build_seed.py`, then `scripts/build_samples.py` |
| `samples/sample.csv`, `labeled.csv`, `empty.csv` | Yes | The demo input, the 17-row accuracy set, an empty file | `scripts/build_samples.py` |
| `reports/` | No (ignored) | CSV output of each run | Created by a run |
| `private/` | No (ignored) | Put your own real statements here | You create it |
| `~/.pennywise/rules.json` | Outside the repo | Merchant rules you teach it by answering questions | Created when you answer one |

### Adding or rebuilding the dataset

The committed seed rules, samples and mock answers are enough to run everything, so you only
need the raw file to rebuild them. To get it, either run the fetch script:

```bash
python scripts/fetch_data.py       # saves data/raw/transactions-synthetic.csv, TLS verified
```

or download it yourself and put it at exactly that path (keep the file name):

```
https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/resolve/3e8d052cb7b362fb1a04b6064d2331982553ea56/transactions-synthetic.csv
```

Then rebuild, **in this order** (the second script reads what the first writes):

```bash
python scripts/build_seed.py       # data/seed_rules.json + data/mock_answers.json
python scripts/build_samples.py    # samples/*.csv + merchant answers for the mock
```

### Using your own statements

Any CSV with the columns `date`, `description`, `amount` works: `pennywise private/march.csv`.
Dates are `YYYY-MM-DD` or `MM/DD/YYYY`; a `[debit]` or `[credit]` prefix on the description sets
the direction (no prefix means debit). Keep real statements in `private/` (ignored by git), not in
`samples/`, which is committed.

## Limits

- One currency (USD) per file; no conversion.
- The plan step is validated and shown, but the pipeline order is fixed: the LLM cannot
  skip or reorder steps. This is deliberate; it keeps a bad model reply from breaking a run.
- The web page cannot pause a run to ask a question. It shows the questions after the run
  and re-runs once you answer.
