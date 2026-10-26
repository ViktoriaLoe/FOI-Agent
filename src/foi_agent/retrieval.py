"""BM25 search over the knowledge base in `foi_agent/data`.

Each source is one JSON file of the form {"source", "licence", "records": [...]}, and each record
is indexed as one document.
"""

import json
import re
from functools import cache
from importlib import resources

from rank_bm25 import BM25Okapi

SOURCES = ("regime_guidance", "exemption_guidance", "disclosure_log", "publication_scheme")

# Fields left out of the index text because they don't describe the content.
_UNINDEXED = {"id", "url"}

# fmt: off
_STOPWORDS = frozenset(
    {
        "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "been", "by", "can",
        "could", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
        "is", "it", "its", "me", "my", "of", "on", "or", "our", "please", "so", "than", "that",
        "the", "their", "them", "there", "these", "this", "those", "to", "us", "was", "we", "were",
        "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
    }
)
# fmt: on


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus common English stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _index_text(record: dict) -> str:
    parts = []
    for key, value in record.items():
        if key in _UNINDEXED:
            continue
        parts.append(" ".join(map(str, value)) if isinstance(value, list) else str(value))
    return " ".join(parts)


@cache
def load_records(source: str) -> tuple[dict, ...]:
    """The records in one source file."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}")
    path = resources.files("foi_agent") / "data" / f"{source}.json"
    return tuple(json.loads(path.read_text(encoding="utf-8"))["records"])


@cache
def _index(source: str) -> BM25Okapi:
    return BM25Okapi([tokenise(_index_text(r)) for r in load_records(source)])


def search(source: str, query: str, k: int = 3) -> list[dict]:
    """Top-k records from `source` for `query`, best first. Records scoring zero are dropped."""
    tokens = tokenise(query)
    if not tokens:
        return []
    records = load_records(source)
    scores = _index(source).get_scores(tokens)
    ranked = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    return [
        {"source": source, "score": float(scores[i]), **records[i]}
        for i in ranked[:k]
        if scores[i] > 0
    ]
