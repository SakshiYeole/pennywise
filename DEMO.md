# Pennywise: 3-minute demo

Record from the web page (`streamlit run app.py`) or the terminal (`pennywise samples/sample.csv`);
both show the same steps. Everything below also works offline with `--mock-llm`
(pick "Offline mock: works" on the page).

## Before you start

1. `python -m pennywise.llm` prints the model and latency. If it fails, use the mock for the whole demo.
2. Start from a clean memory: `rm -f ~/.pennywise/rules.json` (or the page's "Reset saved rules").
3. Open `reports/` somewhere you can show it (a spreadsheet app works best).

## Script

| Time | Do | Say / point at |
| --- | --- | --- |
| 0:00 | Run `samples/sample.csv` | "14 rows, 6 of them traps. The agent reads the file, writes a plan, then every tool call prints a line." |
| 0:30 | Scroll the log | `read_csv` found 2 unreadable rows. `categorize`: 8 by rule, 3 by LLM, 1 unsure. Only unknown merchants go to the model. |
| 0:50 | Flagged tab | The $12,500,000 Delta charge is held out of the totals. The second Netflix charge is `POSSIBLE_DUPLICATE`. The Starbucks row says "ignore previous instructions": flagged `SUSPICIOUS_TEXT`, still categorized Restaurants, totals unchanged. |
| 1:15 | Quarantined tab + the check line | Garbage date and missing amount were quarantined, not guessed. `11 counted + 1 excluded + 2 quarantined = 14 rows read`. |
| 1:30 | Answer the question for `NORTHWIND HOLDINGS 4402` (Shopping) | "The model said 0.30 confidence, so it asked instead of guessing. My answer is saved as a rule." |
| 2:00 | Run again | The question is gone and the log says `9 by rule`; the model now labels 3 merchants instead of 4. "Corrections stick." |
| 2:20 | Open both CSVs from `reports/` | `rows.csv` has every row with category, source, status and flags; `summary.csv` has totals and the reconciliation check. |
| 2:35 | Switch to "Offline mock: outage" (CLI: `--mock-llm=fail`), run | "LLM down: still finishes, unknown merchants are Uncategorized, banner says degraded." |
| 2:50 | Optional: `--confirm 3` | "After review, the big charge can be counted." |

## Reset between takes

```bash
rm -f ~/.pennywise/rules.json reports/*.csv
```
