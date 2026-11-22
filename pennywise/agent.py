"""The agent: read -> plan -> categorize (and ask) -> flag -> totals -> write report.

The LLM only ever sees merchant strings (to label them) and file statistics (to plan).
Everything else, including all money math, is plain code. Each step is announced as an
`Event`, so the terminal and the web page show exactly the same run.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError
from .schema import (
    ALLOWED_TOOLS, CATEGORIES, MAX_QUESTIONS, Event, InputError, Label, merchant_key, money,
)
from .tools import (
    DEFAULT_SEED, RuleStore, categorize, compute_totals, flag_anomalies, read_csv, write_report,
)

MAX_STEPS, MAX_LLM_CALLS, MAX_TOKENS = 8, 5, 20_000

PLAN_SYSTEM = (
    "TASK: PLAN\n"
    "You plan how an agent processes a bank-transaction CSV. The agent's tools, in the order it "
    "must run them, are: " + ", ".join(ALLOWED_TOOLS) + ".\n"
    'Reply with JSON only: {"plan": [<tool names>], "summary": "<one sentence for the user>"}'
)


class Budget:
    """Hard caps so a run can never loop or spend without limit."""

    def __init__(self):
        self.steps = 0
        self.llm_calls = 0
        self.tokens = 0

    def can_call(self, llm) -> bool:
        self.tokens = llm.usage().get("total_tokens", 0)
        return self.llm_calls < MAX_LLM_CALLS and self.tokens < MAX_TOKENS


@dataclass
class RunResult:
    path: str
    rows: list
    quarantined: list
    rows_read: int
    labels: dict
    flags: dict
    excluded_ids: set
    totals: object
    report_paths: object          # (rows_csv, summary_csv) or None
    degraded: bool
    pending: list                 # merchants still waiting for a human answer
    plan: list
    plan_summary: str
    usage: dict
    steps: int
    llm_calls: int
    warnings: list = field(default_factory=list)


def _validate_plan(obj):
    """Accept only a list of known tool names plus a short summary; anything else is ignored."""
    if not isinstance(obj, dict):
        return None
    plan = obj.get("plan")
    if not isinstance(plan, list) or not all(isinstance(t, str) and t in ALLOWED_TOOLS for t in plan):
        return None
    summary = " ".join(str(obj.get("summary") or "").split())[:200]
    return plan, summary


def run(path, out_dir="reports", llm=None, rules_path=None, seed_path=DEFAULT_SEED,
        ask=None, on_event=None, confirmed=()):
    """Process one CSV. `ask(pending, categories) -> (category, remember) | None` is optional."""
    forward = on_event or (lambda e: None)
    budget, store = Budget(), RuleStore(rules_path, seed_path)
    warnings = []
    if llm is not None:
        llm.reset_usage()

    def emit(e):
        if e.kind == "warn":          # keep every warning, wherever it came from
            warnings.append(e.text)
        forward(e)

    def warn(text, tool=""):
        emit(Event("warn", tool, text))

    def step(name, args, fn, counted=True):
        if counted:
            budget.steps += 1
        emit(Event("tool_start", name, args))
        t0 = time.perf_counter()
        try:
            result, summary = fn()
        except InputError as e:
            emit(Event("error", name, str(e)))
            raise
        emit(Event("tool_end", name, summary, {"seconds": time.perf_counter() - t0}))
        return result

    # 1. read_csv ------------------------------------------------------------
    def do_read():
        r = read_csv(path)
        return r, f"{len(r.rows)} rows read, {len(r.quarantined)} quarantined"

    read = step("read_csv", f"file={Path(path).name}", do_read)
    rows, quarantined = read.rows, read.quarantined

    # 2. plan: one LLM call that sees statistics only, never merchant text -----
    state = {"plan": list(ALLOWED_TOOLS), "summary": "", "degraded": False}

    def do_plan():
        known = sum(1 for r in rows if store.lookup(merchant_key(r.merchant), r.direction))
        state["summary"] = (f"{known} of {len(rows)} rows match saved rules; "
                            f"{len(rows) - known} need the LLM. Then flag, total and write the CSV files.")
        if llm is not None and budget.can_call(llm):
            budget.llm_calls += 1
            prompt = (f"The file has {len(rows)} usable rows and {len(quarantined)} unreadable rows. "
                      f"{known} rows match saved merchant rules; {len(rows) - known} do not.")
            try:
                data = llm.chat_json([{"role": "system", "content": PLAN_SYSTEM},
                                      {"role": "user", "content": prompt}])
                ok = _validate_plan(data)
                if ok and ok[1]:
                    state["summary"] = ok[1]
                if ok is None:
                    warn("plan reply was not valid; using the standard plan", "plan")
                elif ok[0] != list(ALLOWED_TOOLS):
                    warn("plan adjusted: required steps always run, in the standard order", "plan")
            except LLMError as e:
                state["degraded"] = True
                warn(f"LLM unavailable ({str(e)[:100]}); running in degraded mode with rules only", "plan")
        return None, "plan ready"

    step("plan", f"rows={len(rows)}", do_plan)
    emit(Event("plan", "plan", state["summary"], {"steps": ["read_csv"] + list(ALLOWED_TOOLS)}))
    live_llm = None if state["degraded"] else llm

    # 3. categorize ----------------------------------------------------------
    def do_cat():
        out = categorize(rows, store, live_llm, budget, emit)
        by_source = {}
        for lab in out[0].values():
            by_source[lab.source] = by_source.get(lab.source, 0) + 1
        text = ", ".join(f"{n} by {s}" for s, n in sorted(by_source.items())) or "nothing to label"
        return out, text

    labels, pending, cat_degraded = step("categorize", f"rows={len(rows)}", do_cat)

    # 3b. ask the user, only for merchants the LLM was unsure about -------------
    remaining, asked = [], 0
    for p in pending:
        if ask is not None and asked < MAX_QUESTIONS:
            asked += 1
            emit(Event("question", "ask_user", p.merchant,
                       {"key": p.key, "suggestion": p.suggestion, "confidence": p.confidence}))
            answer = ask(p, CATEGORIES)
            if answer:
                category, remember = answer
                for i in p.ids:
                    labels[i] = Label(i, category, 1.0, "user", "answered by user")
                if remember:
                    def do_save(p=p, category=category):
                        ok = store.save(p.key, category)
                        return None, "saved for next run" if ok else "kept in memory (disk write failed)"
                    step("save_rule", f"{p.key} -> {category}", do_save, counted=False)
                continue
        remaining.append(p)

    # 4. flag_anomalies ------------------------------------------------------
    def do_flag():
        flags, excluded = flag_anomalies(rows, confirmed)
        kinds = {}
        for fl in flags.values():
            for f in fl:
                kinds[f.flag] = kinds.get(f.flag, 0) + 1
        text = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "nothing flagged"
        return (flags, excluded), text

    flags, excluded_ids = step("flag_anomalies", f"rows={len(rows)}", do_flag)

    # 5. compute_totals ------------------------------------------------------
    def do_totals():
        t = compute_totals(rows, labels, excluded_ids, quarantined, read.rows_read)
        mark = "reconciles" if t.reconciles else "DOES NOT RECONCILE"
        return t, f"spend {money(t.total_spend)}, income {money(t.income)}, net {money(t.net)}; {mark}"

    totals = step("compute_totals", f"counted={len(rows) - len(excluded_ids)}", do_totals)

    # 6. write_report --------------------------------------------------------
    def do_write():
        paths = write_report(Path(path).stem, out_dir, rows, quarantined, labels, flags,
                             excluded_ids, totals, emit)
        return paths, (f"wrote {paths[0].name} and {paths[1].name}" if paths else "not written")

    paths = step("write_report", f"out={out_dir}", do_write)

    return RunResult(
        path=str(path), rows=rows, quarantined=quarantined, rows_read=read.rows_read,
        labels=labels, flags=flags, excluded_ids=excluded_ids, totals=totals,
        report_paths=paths, degraded=state["degraded"] or cat_degraded, pending=remaining,
        plan=state["plan"], plan_summary=state["summary"],
        usage=llm.usage() if llm is not None else {"total_tokens": 0},
        steps=budget.steps, llm_calls=budget.llm_calls, warnings=warnings,
    )


def save_answers(rules_path, answers):
    """Remember (merchant key, category) answers; used by the web page's form."""
    store = RuleStore(rules_path, seed_path=None)
    return [store.save(key, category) for key, category in answers]
