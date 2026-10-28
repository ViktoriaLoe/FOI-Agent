import json
from pathlib import Path

import pytest

from foi_agent.language import is_probably_english

GOLDEN = Path(__file__).parents[1] / "evals" / "datasets" / "golden.jsonl"
POLISH = "Dzień dobry, ile zarabiał dyrektor generalny rady w roku 2025/26 i czy dostał premię?"
WELSH = "Faint o arian a wariodd y cyngor ar ymgynghorwyr allanol yn 2025/26, os gwelwch yn dda?"


@pytest.mark.parametrize("text", [POLISH, WELSH])
def test_other_languages_are_flagged(text):
    assert not is_probably_english(text)


@pytest.mark.parametrize(
    "text",
    [
        "Please send me the report.",
        "What time does Central Library open on Saturdays?",
        "how much is the chief exec paid?? and do they get bonuses on top of that.",
        (
            "We act for Mr Daniel Okafor and enclose his signed authority. Please provide all "
            "records relating to our client's housing benefit claim."
        ),
        (
            "My name is Priya Shah, I live at 14 Hollins Road, NB3 2QT, and you can call me on "
            "07700 900456. Please send me the council's current pay policy statement."
        ),
        (
            "How many penalty charge notices did the council issue in 2024/25, and how many "
            "have been issued so far in 2026/27?"
        ),
    ],
)
def test_english_requests_pass(text):
    assert is_probably_english(text)


@pytest.mark.parametrize("text", ["", "Dzień dobry", "Salaries 2025/26?"])
def test_text_too_short_to_judge_passes(text):
    assert is_probably_english(text)


def test_single_letters_do_not_count_as_english():
    # "a", "i" and "o" are words in Polish and Welsh too.
    assert not is_probably_english("a i o w z y a i o w z y dyrektor rady roku premię")


def test_only_the_polish_golden_request_is_flagged():
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 63
    flagged = [
        r["inputs"]["request"] for r in rows if not is_probably_english(r["inputs"]["request"])
    ]
    assert flagged == [POLISH]
