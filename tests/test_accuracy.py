"""17 hand-labeled rows, one per category, drawn from the open dataset (samples/labeled.csv)."""
import csv

import pytest

from pennywise.agent import run
from pennywise.llm import MockLLM, RealLLM


def _score(samples, tmp_path, llm):
    path = samples / "labeled.csv"
    with path.open(encoding="utf-8", newline="") as f:
        expected = {line: row["expected_category"]
                    for line, row in enumerate(csv.DictReader(f), start=2)}
    result = run(path, out_dir=tmp_path / "out", llm=llm, rules_path=tmp_path / "rules.json", seed_path=None)
    wrong = {line: (expected[line], result.labels[line].category)
             for line in expected if result.labels[line].category != expected[line]}
    return len(expected), len(expected) - len(wrong), wrong


def test_accuracy_offline(samples, tmp_path):
    """Plumbing only: the mock answers with the dataset's own labels."""
    total, correct, wrong = _score(samples, tmp_path, MockLLM("ok"))
    assert total == 17 and correct >= 15, wrong


@pytest.mark.live
def test_accuracy_live(samples, tmp_path):
    """The real number: the configured endpoint labels all 17 rows with no rules to lean on."""
    total, correct, wrong = _score(samples, tmp_path, RealLLM())
    print(f"live accuracy: {correct}/{total}", wrong)
    assert correct >= 15, f"{correct}/{total} correct; misses (line: expected, got): {wrong}"
