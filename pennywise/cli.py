"""Command line entry point: pennywise <file.csv> [options]."""
import argparse
import sys

from rich.console import Console
from rich.markup import escape

from .agent import run
from .llm import LLMError, MockLLM, RealLLM, ping
from .report import print_report, render_event
from .schema import CATEGORIES, InputError

MOCK_MODES = ("ok", "fail", "inject")


def _read(console, prompt):
    """Read one line; a closed input (Ctrl-D, a closed pipe) counts as an empty answer."""
    try:
        return console.input(prompt).strip()
    except EOFError:
        return ""


def _make_ask(console):
    def ask(p, categories):
        console.print(f"\n[bold yellow]?[/] Not sure about [bold]{escape(p.merchant)}[/] "
                      f"(LLM guessed {escape(p.suggestion)}, confidence {p.confidence:.2f}). Pick a category:")
        for i, cat in enumerate(categories, 1):
            console.print(f"  {i:>2}. {cat}")
        choice = _read(console, "Number, or Enter to leave it Uncategorized: ")
        if not choice.isdigit() or not 1 <= int(choice) <= len(categories):
            return None
        remember = _read(console, "Remember this for next time? [Y/n] ").lower() != "n"
        return categories[int(choice) - 1], remember
    return ask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pennywise", description="Categorize a messy bank CSV and write a reconciled report.")
    ap.add_argument("file", nargs="?", help="CSV with columns date,description,amount")
    ap.add_argument("--mock-llm", nargs="?", const="ok", metavar="MODE",
                    help="run offline; MODE is ok (default), fail or inject")
    ap.add_argument("--confirm", type=int, action="append", default=[], metavar="LINE",
                    help="count a flagged row after you reviewed it (repeatable)")
    ap.add_argument("--rules", help="merchant rules file (default ~/.pennywise/rules.json)")
    ap.add_argument("--out", default="reports", help="folder for the CSV files (default reports)")
    ap.add_argument("--ping", action="store_true", help="make one tiny LLM call and print model and latency")
    args = ap.parse_args(argv)

    # `pennywise --mock-llm file.csv` would swallow the file name as the mode; undo that.
    if args.mock_llm is not None and args.mock_llm not in MOCK_MODES:
        if args.file is not None:
            ap.error(f"--mock-llm MODE must be one of {', '.join(MOCK_MODES)}")
        args.file, args.mock_llm = args.mock_llm, "ok"

    console = Console()
    llm = MockLLM(args.mock_llm) if args.mock_llm else RealLLM()

    if args.ping:
        try:
            info = {"mock": True, "mode": args.mock_llm} if args.mock_llm else ping()
        except LLMError as e:
            console.print(f"[red]LLM check failed:[/] {escape(str(e))}")
            return 1
        console.print(info)
        return 0
    if not args.file:
        ap.error("a CSV file is required (or use --ping)")

    ask = _make_ask(console) if sys.stdin.isatty() else None
    try:
        result = run(args.file, out_dir=args.out, llm=llm, rules_path=args.rules, ask=ask,
                     on_event=lambda e: render_event(console, e), confirmed=args.confirm)
    except InputError as e:
        console.print(f"\n[red]error:[/] {escape(str(e))}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/] Nothing was written for this run.")
        return 130
    print_report(console, result)
    return 0
