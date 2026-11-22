"""Build data/seed_rules.json and the key-level part of data/mock_answers.json from the dataset.

Seed rules keep only merchants whose label is unanimous in the dataset and that do not
depend on the debit/credit sign (so Transfer and Income are never seeded).
"""
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from pennywise.schema import clean_description, merchant_key

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "transactions-synthetic.csv"
MUST_HAVE = ("starbucks", "netflix")
SKIP_CATEGORIES = {"Transfer", "Income"}
PER_CATEGORY = 3


def load():
    by_key = defaultdict(Counter)
    with RAW.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            _, text = clean_description(row["description"])
            key = merchant_key(text)
            if key:
                by_key[key][row["category"]] += 1
    return by_key


def main():
    by_key = load()
    pure = {}      # key -> (category, count) where the label is unanimous
    mostly = {}    # key -> category where one label covers >= 90%
    for key, counts in by_key.items():
        (cat, n), total = counts.most_common(1)[0], sum(counts.values())
        if n == total and total >= 8:
            pure[key] = (cat, total)
        if total >= 3 and n / total >= 0.9:
            mostly[key] = cat

    chosen, per_cat = {}, Counter()
    for key in MUST_HAVE:
        if key in pure:
            chosen[key] = pure[key][0]
            per_cat[pure[key][0]] += 1
    for key, (cat, total) in sorted(pure.items(), key=lambda kv: -kv[1][1]):
        if cat in SKIP_CATEGORIES or len(key) < 4 or key in chosen or per_cat[cat] >= PER_CATEGORY:
            continue
        chosen[key] = cat
        per_cat[cat] += 1

    today = date.today().isoformat()
    seed = {"version": 1, "rules": {k: {"category": c, "source": "seed", "updated": today}
                                    for k, c in sorted(chosen.items())}}
    (ROOT / "data" / "seed_rules.json").write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")

    answers = {"by_key": {k: {"category": c, "confidence": 0.9} for k, c in sorted(mostly.items())},
               "by_text": {}}
    (ROOT / "data" / "mock_answers.json").write_text(json.dumps(answers, indent=1) + "\n", encoding="utf-8")
    missing = [k for k in MUST_HAVE if k not in chosen]
    print(f"seed rules: {len(chosen)} ({dict(per_cat)}); mock key answers: {len(mostly)}"
          + (f"; NOT pure enough to seed: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
