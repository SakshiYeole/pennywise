"""Pennywise web page: watch the agent work, then read the report.

Run:  streamlit run app.py
It shows the same event stream as the terminal, so a screen recording of this page is a
faithful recording of the agent. Text that came from the CSV is only ever shown with
st.code / st.dataframe, never as markdown.
"""
import re
import tempfile
from pathlib import Path

import streamlit as st

from pennywise.agent import run, save_answers
from pennywise.llm import MockLLM, RealLLM
from pennywise.report import event_line, footer_text
from pennywise.schema import CATEGORIES, InputError, money
from pennywise.tools import DEFAULT_RULES

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 5_000_000
LLM_CHOICES = {
    "Live endpoint (from .env)": None,
    "Offline mock: works": "ok",
    "Offline mock: outage": "fail",
    "Offline mock: hijacked model": "inject",
}

st.set_page_config(page_title="Pennywise", page_icon="\U0001FA99", layout="wide")
st.title("Pennywise")
st.caption("A terminal agent for messy bank CSVs, shown step by step. "
           "It categorizes, flags what looks wrong, and writes a report that reconciles to the file.")

# ---- sidebar: what to run -----------------------------------------------------
with st.sidebar:
    st.header("Input")
    source = st.radio("CSV file", ["Sample file", "Upload a CSV"], label_visibility="collapsed")
    uploaded, sample_name = None, None
    if source == "Sample file":
        samples = sorted(p.name for p in (ROOT / "samples").glob("*.csv"))
        sample_name = st.selectbox("Sample", samples, index=samples.index("sample.csv") if "sample.csv" in samples else 0)
    else:
        uploaded = st.file_uploader("Columns: date, description, amount", type=["csv"])

    st.header("LLM")
    llm_choice = st.selectbox("Model", list(LLM_CHOICES), label_visibility="collapsed")
    rules_path = st.text_input("Saved merchant rules", str(DEFAULT_RULES))
    confirm_text = st.text_input("Count flagged lines after review", placeholder="e.g. 3, 9",
                                 help="Line numbers from the Flagged tab.")
    run_clicked = st.button("Run the agent", type="primary", width="stretch")

    with st.expander("Reset saved rules"):
        sure = st.checkbox("Delete the rules file above")
        if st.button("Reset", disabled=not sure):
            Path(rules_path).expanduser().unlink(missing_ok=True)
            st.success("Saved rules cleared.")

log_box = st.container()
result_box = st.container()


# ---- running the agent --------------------------------------------------------
def make_llm(label):
    mode = LLM_CHOICES[label]
    return RealLLM() if mode is None else MockLLM(mode)


def stage_input():
    """Return (path, error). Uploaded files go to a temp folder under a sanitized name."""
    if uploaded is not None:
        data = uploaded.getvalue()
        if len(data) > MAX_UPLOAD_BYTES:
            return None, "That file is over 5 MB; Pennywise handles statements, not databases."
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(uploaded.name).name) or "upload.csv"
        path = Path(tempfile.mkdtemp(prefix="pennywise-")) / name
        path.write_bytes(data)
        return path, None
    if sample_name:
        return ROOT / "samples" / sample_name, None
    return None, "Choose a sample or upload a CSV first."


def parse_confirm(text):
    return [int(t) for t in re.findall(r"\d+", text or "")]


def execute(params):
    """Run the agent, streaming its events into a live log; keep everything in session state."""
    lines, st.session_state["result"], st.session_state["error"] = [], None, None
    with log_box:
        with st.status("Agent running…", expanded=True) as status:
            live = st.empty()

            def on_event(e):
                line = event_line(e)
                if line:
                    lines.append(line)
                    live.code("\n".join(lines), language=None, wrap_lines=True)

            try:
                st.session_state["result"] = run(
                    params["path"], out_dir=ROOT / "reports", llm=make_llm(params["llm"]),
                    rules_path=Path(params["rules"]).expanduser(), on_event=on_event,
                    confirmed=params["confirmed"],
                )
                status.update(label="Agent finished", state="complete")
            except InputError as e:
                st.session_state["error"] = str(e)
                status.update(label="Agent stopped", state="error")
    st.session_state["lines"] = lines
    st.session_state["params"] = params
    st.rerun()


if run_clicked:
    path, problem = stage_input()
    if problem:
        st.session_state["error"], st.session_state["result"], st.session_state["lines"] = problem, None, []
    else:
        execute({"path": path, "llm": llm_choice, "rules": rules_path, "confirmed": parse_confirm(confirm_text)})

# ---- showing what happened ----------------------------------------------------
lines, result, error = st.session_state.get("lines"), st.session_state.get("result"), st.session_state.get("error")

if lines:
    with log_box:
        with st.expander("Agent log (every tool call)", expanded=True):
            st.code("\n".join(lines), language=None, wrap_lines=True)
if error:
    result_box.error(error)
if result is None and not error:
    result_box.info("Pick an input on the left and press **Run the agent**. "
                    "Try the sample file with the offline mock first.")

if result is not None:
    t = result.totals
    with result_box:
        if result.degraded:
            st.warning("Degraded mode: the LLM was unavailable, so unknown merchants are Uncategorized. "
                       "Rules-only results are still correct.")

        cols = st.columns(5)
        cols[0].metric("Rows read", t.rows_read)
        cols[1].metric("Counted", t.counted)
        cols[2].metric("Needs review", t.excluded)
        cols[3].metric("Quarantined", t.quarantined)
        cols[4].metric("Net", money(t.net))
        check = (f"{t.counted} counted + {t.excluded} excluded + {t.quarantined} quarantined "
                 f"= {t.rows_read} rows read")
        (st.success if t.reconciles else st.error)(("Reconciles: " if t.reconciles else "Does not reconcile: ") + check)

        totals_tab, flagged_tab, quarantine_tab, rows_tab = st.tabs(["Totals", "Flagged", "Quarantined", "All rows"])
        with totals_tab:
            table = [{"Category": c, "Rows": t.counts.get(c, 0), "Total": money(a)} for c, a in t.spend.items()]
            if "Income" in t.counts:
                table.append({"Category": "Income", "Rows": t.counts["Income"], "Total": money(t.income)})
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption(f"Spend {money(t.total_spend)} · Income {money(t.income)} · Net {money(t.net)}")
        with flagged_tab:
            flagged = [{"Line": r.line, "Description": r.merchant, "Amount": money(r.amount),
                        "Flags": ", ".join(f.flag for f in result.flags[r.id]),
                        "Counted": "no, needs review" if r.id in result.excluded_ids else "yes"}
                       for r in result.rows if r.id in result.flags]
            if flagged:
                st.dataframe(flagged, hide_index=True, width="stretch")
                if result.excluded_ids:
                    st.caption("Excluded rows stay out of the totals until you list their line numbers "
                               "in the sidebar and run again.")
            else:
                st.write("Nothing flagged.")
        with quarantine_tab:
            if result.quarantined:
                st.dataframe([{"Line": q.line, "Reason": q.reason, "Raw text": q.raw} for q in result.quarantined],
                             hide_index=True, width="stretch")
            else:
                st.write("No unreadable rows.")
        with rows_tab:
            st.dataframe([{"Line": r.line, "Date": r.date, "Description": r.merchant, "Direction": r.direction,
                           "Amount": f"{r.amount:.2f}", "Category": result.labels[r.id].category,
                           "Source": result.labels[r.id].source, "Confidence": round(result.labels[r.id].confidence, 2),
                           "Status": "excluded" if r.id in result.excluded_ids else "counted"}
                          for r in result.rows], hide_index=True, width="stretch")

        if result.report_paths:
            st.subheader("CSV report")
            d1, d2 = st.columns(2)
            for col, path, label in ((d1, result.report_paths[0], "Download rows CSV"),
                                     (d2, result.report_paths[1], "Download summary CSV")):
                col.download_button(label, Path(path).read_bytes(), file_name=Path(path).name,
                                    mime="text/csv", width="stretch")

        if result.pending:
            st.subheader("Needs your answer")
            st.caption("The model was unsure about these merchants. Your answers are saved as rules, "
                       "so the next run needs no LLM call for them.")
            with st.form("answers"):
                picks = {}
                for p in result.pending:
                    picks[p.key] = st.selectbox(
                        f"{p.merchant}  (model guessed {p.suggestion}, confidence {p.confidence:.2f})",
                        ["(leave Uncategorized)"] + list(CATEGORIES), key=f"pick-{p.key}")
                if st.form_submit_button("Save as rules and run again", type="primary"):
                    chosen = [(k, v) for k, v in picks.items() if v in CATEGORIES]
                    if chosen:
                        save_answers(Path(st.session_state["params"]["rules"]).expanduser(), chosen)
                        execute(st.session_state["params"])

        st.caption(footer_text(result))
