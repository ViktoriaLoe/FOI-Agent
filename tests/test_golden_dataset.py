"""Catches labelling slips in the golden dataset before they cost an eval run."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

from foi_agent.retrieval import SOURCES, load_records

DATASET = Path(__file__).parent.parent / "evals" / "datasets" / "golden.jsonl"
EXAMPLES = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
CORPUS_IDS = {record["id"] for source in SOURCES for record in load_records(source)}
OUTPUT_KEYS = {"regime", "action", "published_id", "exemptions", "relevant_ids", "deadline"}


def label(example):
    return f"{example['metadata']['slice']}: {example['inputs']['request'][:50]}"


@pytest.mark.parametrize("example", EXAMPLES, ids=label)
def test_example_is_well_formed(example):
    inputs, outputs, metadata = example["inputs"], example["outputs"], example["metadata"]

    assert set(inputs) == {"request", "received"}
    date.fromisoformat(inputs["received"])
    assert set(outputs) <= OUTPUT_KEYS
    assert outputs["action"] in {"send", "escalate", "route", "clarify"}
    assert outputs.get("regime", "FOI") in {"FOI", "EIR", "SAR", "BAU"}
    assert metadata["category"] in {"normal", "edge", "high_risk"}
    assert metadata["split"] in {"dev", "test"}
    assert metadata["why"].strip()


@pytest.mark.parametrize("example", EXAMPLES, ids=label)
def test_labels_agree_with_each_other(example):
    outputs = example["outputs"]
    if outputs["action"] == "send":
        # The agent may only send a signpost to something published.
        assert outputs["regime"] == "FOI" and outputs["published_id"]
    if outputs["action"] == "route":
        assert outputs["regime"] in {"SAR", "BAU"}
    if outputs.get("regime") in {"SAR", "BAU"}:
        assert "deadline" not in outputs  # FOI deadlines don't apply


@pytest.mark.parametrize("example", EXAMPLES, ids=label)
def test_every_labelled_id_exists_in_the_corpus(example):
    outputs = example["outputs"]
    ids = set(outputs.get("relevant_ids") or []) | {outputs.get("published_id")} - {None}
    assert ids <= CORPUS_IDS


@pytest.mark.parametrize("example", EXAMPLES, ids=label)
def test_exemptions_are_written_as_sections(example):
    for section in example["outputs"].get("exemptions") or []:
        assert re.fullmatch(r"s\d+[A-Z]?", section)


@pytest.mark.parametrize("example", [e for e in EXAMPLES if "deadline" in e["outputs"]], ids=label)
def test_deadline_is_a_weekday_after_receipt(example):
    received = date.fromisoformat(example["inputs"]["received"])
    deadline = date.fromisoformat(example["outputs"]["deadline"])
    assert deadline.weekday() < 5
    assert 26 <= (deadline - received).days <= 40  # 20 working days, plus any holidays


def test_both_splits_cover_every_category():
    for split in ("dev", "test"):
        categories = {
            e["metadata"]["category"] for e in EXAMPLES if e["metadata"]["split"] == split
        }
        assert categories == {"normal", "edge", "high_risk"}
