"""BM25 search over the knowledge base in `foi_agent/data`.

Each source is one JSON file of the form {"source", "licence", "records": [...]}, and each record
is indexed as one document.
"""

import hashlib
import json
import re
from functools import cache
from importlib import resources

from langsmith import traceable
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


def _source_path(source: str):
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}")
    return resources.files("foi_agent") / "data" / f"{source}.json"


@cache
def load_records(source: str) -> tuple[dict, ...]:
    """The records in one source file."""
    return tuple(json.loads(_source_path(source).read_text(encoding="utf-8"))["records"])


@cache
def corpus_version() -> str:
    """Short hash of the knowledge base, so traces and experiments record which data they used."""
    digest = hashlib.sha256()
    for source in SOURCES:
        digest.update(_source_path(source).read_bytes())
    return digest.hexdigest()[:12]


@cache
def _index(source: str) -> BM25Okapi:
    return BM25Okapi([tokenise(_index_text(r)) for r in load_records(source)])


def _as_trace_documents(hits: list[dict]) -> dict:
    """Show hits in LangSmith as retriever documents, with id, score and URL as metadata."""
    documents = []
    for hit in hits:
        record = {k: v for k, v in hit.items() if k not in {"source", "score"}}
        metadata = {k: hit[k] for k in ("id", "source", "score", "url") if k in hit}
        documents.append(
            {"type": "Document", "page_content": _index_text(record), "metadata": metadata}
        )
    return {"documents": documents}


@traceable(run_type="retriever", name="bm25_search", process_outputs=_as_trace_documents)
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
