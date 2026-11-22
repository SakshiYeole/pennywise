"""Shared types, constants and the two text helpers every module agrees on."""
import re
from dataclasses import dataclass, field
from decimal import Decimal

# The 17 labels of the open dataset the demo data comes from, plus a fallback
# that only code assigns (the LLM is never allowed to return it).
CATEGORIES = (
    "Restaurants", "Groceries", "Shopping", "Transportation", "Entertainment",
    "Utilities", "Subscription", "Healthcare", "Insurance", "Mortgage", "Rent",
    "Travel", "Education", "Personal Care", "Transfer", "Income", "Fees",
)
FALLBACK = "Uncategorized"
ALL_CATEGORIES = CATEGORIES + (FALLBACK,)

ASK_BELOW = 0.6          # LLM confidence under this -> ask the user
MAX_QUESTIONS = 5        # per run
BATCH_SIZE = 20          # unknown merchants per LLM call
MAX_MERCHANT_CHARS = 120
ALLOWED_TOOLS = ("categorize", "flag_anomalies", "compute_totals", "write_report")


class InputError(Exception):
    """The input file cannot be processed at all (empty, binary, wrong columns)."""


@dataclass(frozen=True)
class Txn:
    id: int              # the file line number, so a user can refer to it
    line: int
    date: str            # ISO yyyy-mm-dd
    merchant: str        # cleaned description, at most 120 chars
    amount: Decimal      # magnitude as written in the file
    direction: str       # "debit" or "credit"
    currency: str = "USD"


@dataclass
class Quarantined:
    line: int
    raw: str
    reason: str


@dataclass
class Label:
    id: int
    category: str
    confidence: float
    source: str          # rule | llm | user | fallback
    note: str = ""


@dataclass
class Flag:
    id: int
    flag: str            # NEEDS_REVIEW | POSSIBLE_DUPLICATE | SUSPICIOUS_TEXT | CONFIRMED
    reason: str


@dataclass
class Pending:
    """A merchant the LLM was unsure about; one question per merchant."""
    key: str
    merchant: str
    suggestion: str
    confidence: float
    ids: list = field(default_factory=list)


@dataclass
class Totals:
    spend: dict          # category -> net spend (debits minus credits), Decimal
    counts: dict         # category -> number of counted rows
    income: Decimal
    total_spend: Decimal
    net: Decimal
    counted: int
    excluded: int
    excluded_amount: Decimal
    quarantined: int
    rows_read: int
    reconciles: bool


@dataclass
class Event:
    """One visible step of the run; the terminal and the web page both render these."""
    kind: str            # tool_start | tool_end | plan | info | warn | question | error
    tool: str = ""
    text: str = ""
    data: dict = field(default_factory=dict)


def money(d) -> str:
    return f"-${abs(d):,.2f}" if d < 0 else f"${d:,.2f}"


# --- text helpers ------------------------------------------------------------
_SIGN = re.compile(r"^\s*\[(debit|credit)\]\s*", re.I)
_PREFIX = re.compile(
    r"^(?:sq \*|tst\*|clv\*|pp\*|pypl\*|paypal \*"
    r"|preapproved payment bill user payment:|preapproved payment:"
    r"|express checkout payment:|withdrawal from|preauthorized deposit from)\s*",
    re.I,
)


def clean_description(raw: str):
    """Split '[debit] TEXT' into (direction, cleaned text). No prefix means debit."""
    raw = raw or ""
    m = _SIGN.match(raw)
    direction = m.group(1).lower() if m else "debit"
    text = _SIGN.sub("", raw, count=1)
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    text = " ".join(text.split())
    return direction, text[:MAX_MERCHANT_CHARS]


def merchant_key(text: str) -> str:
    """Stable key for rules: POS/PayPal prefixes and any token with a digit dropped,
    first three tokens kept ('TST*GOLDEN DRAGON - OAKLAND' -> 'golden dragon oakland')."""
    t = _PREFIX.sub("", (text or "").strip())
    tokens = [tok for tok in re.split(r"[^a-z0-9&]+", t.lower())
              if tok and tok != "&" and not any(c.isdigit() for c in tok)]
    return " ".join(tokens[:3])
