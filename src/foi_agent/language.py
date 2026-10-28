"""A rough check that a request is in English, so others can be escalated.

The classifier and BM25 retrieval only work on English. This needs no model or dependency: English
text is full of short function words ("the", "of", "how"), and other languages mostly lack them.
"""

import re

# Common English function words. Single letters ("a", "i") are left out, because they are words
# in many languages ("i" is Polish for "and").
FUNCTION_WORDS = frozenset(
    {
        "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
        "been", "before", "being", "both", "but", "by", "can", "could", "did", "do", "does",
        "each", "every", "for", "from", "had", "has", "have", "he", "her", "here", "him", "his",
        "how", "if", "in", "into", "is", "it", "its", "me", "more", "most", "much", "my", "no",
        "not", "of", "on", "or", "our", "please", "she", "should", "so", "some", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "under",
        "until", "up", "us", "was", "we", "were", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "will", "with", "would", "you", "your",
    }
)  # fmt: skip

# The share of function words below which text is probably not English. Every English request
# in the golden dataset scores over 0.25, and the Polish one scores 0.
MIN_SHARE = 0.15
MIN_WORDS = 4  # fewer words than this can't be judged, so they pass

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def is_probably_english(text: str) -> bool:
    """True unless the text is long enough to judge and has too few English function words."""
    words = _WORD.findall(text.lower())
    if len(words) < MIN_WORDS:
        return True
    share = sum(w in FUNCTION_WORDS for w in words) / len(words)
    return share >= MIN_SHARE
