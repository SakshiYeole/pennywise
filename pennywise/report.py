"""Terminal rendering of events and of the finished report."""
from rich.markup import escape
from rich.table import Table

from .agent import MAX_LLM_CALLS, MAX_STEPS
from .schema import money


def event_line(e):
    """Plain-text version of an event (no markup), used by the web page's live log."""
    if e.kind == "tool_start":
        return f"→ {e.tool}({e.text})"
    if e.kind == "tool_end":
        return f"  ✓ {e.text}  {e.data.get('seconds', 0):.1f}s"
    if e.kind == "plan":
        return f"Plan: {e.text}\n  " + " → ".join(e.data.get("steps", []))
    if e.kind == "info":
        return f"  {e.text}"
    if e.kind == "warn":
        return f"  ! {e.text}"
    if e.kind == "error":
        return f"  ✗ {e.text}"
    return None


def tokens_text(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def footer_text(result) -> str:
    tokens = result.usage.get("total_tokens", 0)
    return (f"steps {result.steps}/{MAX_STEPS} · llm calls {result.llm_calls}/{MAX_LLM_CALLS}"
            f" · tokens {tokens_text(tokens)}")


def render_event(console, e):
    """One printed line per event. All text from the file goes through escape()."""
    if e.kind == "tool_start":
        console.print(f"[bold]→ {escape(e.tool)}[/]({escape(e.text)})", soft_wrap=True)
    elif e.kind == "tool_end":
        console.print(f"  [green]✓[/] {escape(e.text)}  [dim]{e.data.get('seconds', 0):.1f}s[/]")
    elif e.kind == "plan":
        console.print(f"[bold cyan]Plan:[/] {escape(e.text)}")
        console.print("[cyan]  " + " → ".join(e.data.get("steps", [])) + "[/]")
    elif e.kind == "info":
        console.print(f"  [dim]{escape(e.text)}[/]")
    elif e.kind == "warn":
        console.print(f"  [yellow]! {escape(e.text)}[/]")
    elif e.kind == "error":
        console.print(f"  [red]✗ {escape(e.text)}[/]")


def print_report(console, result):
    t = result.totals
    console.print()
    if result.degraded:
        console.print("[bold yellow]Degraded mode:[/] the LLM was unavailable, so unknown merchants "
                      "are Uncategorized. Rules-only results are still correct.")

    cats = Table(title="Spending by category", title_justify="left")
    cats.add_column("Category")
    cats.add_column("Rows", justify="right")
    cats.add_column("Total", justify="right")
    for cat, amount in t.spend.items():
        cats.add_row(escape(cat), str(t.counts.get(cat, 0)), money(amount))
    if "Income" in t.counts:
        cats.add_row("Income", str(t.counts["Income"]), money(t.income))
    console.print(cats)

    flagged = [r for r in result.rows if r.id in result.flags]
    if flagged:
        table = Table(title="Flagged rows", title_justify="left")
        for col in ("Line", "Description", "Amount", "Flags", "Counted?"):
            table.add_column(col, overflow="fold", no_wrap=col in ("Line", "Amount", "Flags"))
        for r in flagged:
            names = ", ".join(f.flag for f in result.flags[r.id])
            counted = "no, needs review" if r.id in result.excluded_ids else "yes"
            table.add_row(str(r.line), escape(r.merchant), money(r.amount), names, counted)
        console.print(table)
        if result.excluded_ids:
            lines = " ".join(f"--confirm {i}" for i in sorted(result.excluded_ids))
            console.print(f"[dim]Excluded rows stay out of totals until confirmed, e.g. {lines}[/]")

    if result.quarantined:
        table = Table(title="Quarantined rows (could not be read)", title_justify="left")
        for col in ("Line", "Reason", "Raw text"):
            table.add_column(col, overflow="fold")
        for q in result.quarantined:
            table.add_row(str(q.line), escape(q.reason), escape(q.raw))
        console.print(table)

    if result.pending:
        names = ", ".join(escape(p.merchant) for p in result.pending)
        console.print(f"[yellow]Needs your answer ({len(result.pending)}):[/] {names}")

    console.print()
    console.print(f"Spend {money(t.total_spend)}   Income {money(t.income)}   Net {money(t.net)}")
    mark = "[green]✓ reconciles[/]" if t.reconciles else "[bold red]✗ DOES NOT RECONCILE[/]"
    console.print(f"{mark}: {t.counted} counted + {t.excluded} excluded + {t.quarantined} "
                  f"quarantined = {t.rows_read} rows read")
    if result.report_paths:
        console.print(f"CSV report: {escape(str(result.report_paths[0]))}", soft_wrap=True)
        console.print(f"            {escape(str(result.report_paths[1]))}", soft_wrap=True)
    console.print(f"[dim]{footer_text(result)}[/]")
