from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def samples():
    return ROOT / "samples"


@pytest.fixture
def write_csv(tmp_path):
    def _write(text: str, name="input.csv") -> Path:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p
    return _write


@pytest.fixture
def run_agent(tmp_path):
    """Run the agent offline with an empty rules file and no seed unless asked."""
    from pennywise.agent import run
    from pennywise.llm import MockLLM

    def _run(path, mode="ok", seed=None, ask=None, confirmed=(), rules=None, events=None):
        return run(
            path, out_dir=tmp_path / "out", llm=MockLLM(mode),
            rules_path=rules or tmp_path / "rules.json", seed_path=seed,
            ask=ask, confirmed=confirmed, on_event=events.append if events is not None else None,
        )
    return _run
