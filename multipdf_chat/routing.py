"""Post-retrieval routing: answer / refuse / flag for human."""
import os 
from typing import Literal, Optional
from enum import Enum
from decimal import Decimal
from dataclasses import dataclass, field
from multipdf_chat.retrieval import RetrievedParent
import re

DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("ROUTING_DISTANCE_THRESHOLD", "0.35"))
DEFAULT_MARGIN_THRESHOLD = float(os.getenv("ROUTING_MARGIN_THRESHOLD", "0.04"))
SHORT_QUERY_TOKEN_LIMIT = 3

Status = Literal["answered", "insufficient_evidence", "conflicting_sources"]

class ReasonCode(str, Enum):
    NO_CANDIDATES = "NO_CANDIDATES"
    TOP_DISTANCE_HIGH = "TOP_DISTANCE_HIGH"
    LOW_MARGIN = "LOW_MARGIN"
    SHORT_QUERY_NO_FILTER = "SHORT_QUERY_NO_FILTER"
    FIGURE_CONFLICT = "FIGURE_CONFLICT"
    OK = "OK"

@dataclass(frozen=True)
class Figure:
    """A typed, normalised numeric fact with its lexical context."""
    kind: Literal["money", "pct", "duration_days", "bps"]
    value: Decimal # canonical magnitude (USD, %, days, or bps)
    cluster: str # nearest fee/concept noun phrase, lowercased

    def key(self) -> tuple:
        # 2-decimal rounding so $7 and $7.00 compare equal.
        return (self.kind, self.cluster, self.value.quantize(Decimal("0.01")))

@dataclass
class RoutingDecision:
    status: Status
    reason_code: ReasonCode
    reason: str
    parents: list[RetrievedParent] = field(default_factory=list)
    features: dict = field(default_factory=dict)

# --- Figure extraction ------------------------------------------------------

_MONEY_RE = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?"
)

_PCT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s%"
)

_BPS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s(?:bps|basis\s+points)",
    re.IGNORECASE
)

_DAYS_RE = re.compile(r"(\d+)\s(?:calendar\s+)?day(?:s)?", re.IGNORECASE)

# Noun phrases we care about — extend as the corpus grows.
_CLUSTER_VOCAB = (
    "late fee", "activation fee", "early termination fee", "etf",
    "returned payment fee", "restocking fee", "cancellation fee",
    "interest", "apr", "dispute window", "return window",
)

def _nearest_cluster(text: str, span_start: int, window: int = 60) -> str:
    """Lowercased noun phrase from a window around the match; '' if none."""
    lo = max(0, span_start - window)
    hi = min(len(text), span_start + window)
    hay = text[lo:hi].lower()
    for phrase in _CLUSTER_VOCAB:
        if phrase in hay:
            return phrase
    return ""

def _money(raw_int: str, raw_cents: Optional[str]) -> Decimal:
    whole = Decimal(raw_int.replace(",", ""))
    cents = Decimal(raw_cents) / Decimal(100) if raw_cents else Decimal(0)
    return whole + cents

def extract_figures(text: str) -> list[Figure]:
    out: list[Figure] = []
    for m in _MONEY_RE.finditer(text):
        out.append(
            Figure(
                "money", 
                _money(m.group(1), m.group(2)),
                _nearest_cluster(text, m.start())
            )
        )
    for m in _PCT_RE.finditer(text):
        out.append(
            Figure(
                "pct", 
                Decimal(m.group(1)),
                _nearest_cluster(text, m.start())
            )
        )
    for m in _DAYS_RE.finditer(text):
        out.append(
            Figure(
                "duration_days", 
                Decimal(m.group(1)),
                _nearest_cluster(text, m.start())
            )
        )
    for m in _BPS_RE.finditer(text):
        out.append(
            Figure(
                "bps", 
                Decimal(m.group(1)),
                _nearest_cluster(text, m.start())
            )
        )
    return out

# --- Conflict detection ------------------------------------------------------

def _figures_conflict(parents: list[RetrievedParent], top_n: int = 3) -> Optional[str]:
    """
    Flag a conflict when distinct docs produce >1 distinct canonical value
    for the SAME (kind, cluster). Equivalence (e.g. $7 vs $7.00) is handled
    by the Figure.key() rounding.
    """
    seen_docs: set[str] = set()
    # (kind, cluster) -> {canonical_value: {doc_slug, ...}}
    # ("money", "late fee") → 7.00 : {"doc_A", "doc_B"}  10.00 : {"doc_C"}
    by_bucket: dict[tuple[str, str], dict[Decimal, set[str]]] = {}

    for p in parents[:top_n]:
        if p.doc_slug in seen_docs:
            continue
        seen_docs.add(p.doc_slug)
        for fig in extract_figures(p.content):
            if not fig.cluster:
                continue 
            # Give the dictionary for this type of figure and concept. 
            # If it doesn't exist, create an empty one
            bucket = by_bucket.setdefault((fig.kind, fig.cluster), {})
            # fig.key() -> ("money", "late fee", Decimal("7.00"))
            # Are there docs containing 7.00? If not, return empty set
            bucket.setdefault(fig.key()[2], set()).add(p.doc_slug)

    """
    by_bucket = {
        ("money", "late fee"): {
            Decimal("7.00"): {"doc_A", "doc_B"},
            Decimal("10.00"): {"doc_C"}
        }
    }
    """
    for (kind, cluster), values in by_bucket.items():
        if len(values) >= 2 and sum(len(docs) for docs in values.values()) >= 2:
            # "7.00 (doc_A/doc_B), 10.00 (doc_C)"
            summary = ", ".join(
                f"{v} ({'/'.join(sorted(d))})" for v, d in values.items()
            )
            return f"{cluster} ({kind}) diverges: {summary}"

    return None 

# --- Router ------------------------------------------------------------------
def route(
    question: str,
    parents: list[RetrievedParent],
    product_line: Optional[str],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> RoutingDecision:
    features: dict = {
        "n_candidates": len(parents),
        "product_line": product_line,
        "query_tokens": len(question.split()),
    }
    if not parents:
        return RoutingDecision(
            "insufficient_evidence",
            ReasonCode.NO_CANDIDATES,
            "no candidates returned", 
            features=features
        )
    
    top = parents[0]
    features["top_distance"] = top.distance
    features["top_doc"] = top.doc_slug

    margin = (parents[1].distance - top.distance) if len(parents) > 1 else 1.0
    features["margin"] = margin

    # Short query without a product_line filter is almost always ambiguous —
    # force clarification rather than guess.
    if (features["query_tokens"] <= SHORT_QUERY_TOKEN_LIMIT and product_line is None):
        return RoutingDecision(
            "insufficient_evidence", ReasonCode.SHORT_QUERY_NO_FILTER,
            f"short query ({features['query_tokens']} tokens) with no product_line",
            parents, features,
        )      

    if top.distance > distance_threshold:
        return RoutingDecision(
            "insufficient_evidence",
            ReasonCode.TOP_DISTANCE_HIGH,
            f"top distance {top.distance:.3f} > threshold {distance_threshold}",
            parents,
            features
        )

    if margin < margin_threshold and product_line is None:
        return RoutingDecision(
            "insufficient_evidence", 
            ReasonCode.LOW_MARGIN,
            f"top-1 vs top-2 margin {margin:.3f} < {margin_threshold}",
            parents, 
            features
        )

    # Only look for conflicts when the caller didn't pin a product_line —
    # with a filter in place, the doc set is already narrowed.
    if product_line is None:
        conflict = _figures_conflict(parents)
        if conflict:
            return RoutingDecision(
                "conflicting_sources",
                ReasonCode.FIGURE_CONFLICT,
                conflict, 
                parents, 
                features
            )

    return RoutingDecision("answered", ReasonCode.OK, "", parents, features)


            