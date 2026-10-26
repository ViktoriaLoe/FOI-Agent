# Golden dataset

[golden.jsonl](golden.jsonl) holds 63 FOI requests to the fictional Northbrook Borough Council, each labelled with what an FOI officer would want the agent to do. Labels describe the ideal outcome, not what the current version can do. [tests/test_golden_dataset.py](../../tests/test_golden_dataset.py) checks the format, and that every labelled id exists in the knowledge base.

## Format

```json
{"inputs": {"request": "...", "received": "2026-11-02"},
 "outputs": {"regime": "FOI", "action": "send", "decision": "signpost", "published_id": "PS-003",
             "exemptions": ["s21"], "relevant_ids": ["PS-003", "DL-2026-016"], "deadline": "2026-12-01"},
 "metadata": {"slice": "vocab_mismatch", "category": "edge", "split": "dev", "why": "...",
              "tags": ["bank_holiday"]}}
```

- `action`: `send`, `route` (SAR or business as usual), `escalate` (to an officer) or `clarify` (ask the requester to narrow the request).
- `decision` (FOI only): `provide`, `signpost`, `part_refuse`, `refuse`, `clarify` or `not_held`.
- `published_id`: the record that fully answers the request, a list of records that each do, or `null`.
- `relevant_ids`: records a good search should find.
- `deadline`: 20 working days under FOIA s10, excluding bank holidays in any part of the UK.

A field is left out when it doesn't apply, and its evaluator skips that example. `why` explains each label.

## Slices

`category` follows the brief: normal cases (16), edge cases (27) and failures that would matter in the real world (`high_risk`, 20). Each slice tests one part of the agent, so a failure points to where to look:

| Slice | Examples | Tests | Look at |
| --- | --- | --- | --- |
| `routine` | 6 | Plain SAR, business as usual and EIR | `classify_regime` |
| `regime_trap` | 12 | SARs and EIR worded as FOI, mixed requests | `classify_regime` and `regime_scores` |
| `published` | 9 | Signposting published information, including where escalating would be over-cautious | `bm25_search` spans, then `assess` |
| `vocab_mismatch` | 5 | Published, but in different words or another language | `bm25_search` spans |
| `partly_published` | 4 | Only part of the request, or the wrong period, is published | `assess` |
| `exemption` | 8 | Exemptions the request doesn't name, and precedents | Retrieved guidance, then `assess` |
| `plain_provide` | 5 | Answerable, but nothing published | `assess` and the draft |
| `not_held` | 3 | Information the council doesn't hold | `assess` decision |
| `too_broad` | 3 | Requests that need narrowing | `assess` decision |
| `adversarial` | 4 | Prompt injection and abusive requests | `decide_action` |
| `unsafe_send_bait` | 4 | Requests that tempt an automatic send that would be wrong | `decide_action` |

## Keeping the before and after honest

- **Labels came first.** They were drafted before the agent was run on any example, and the dataset is committed before the baseline.
- **Known gaps are declared.** When the dataset was written, v1 had no `clarify` or `not_held` outcome (6 examples, marked `known_gap`), and its deadline skipped weekends only (34 examples, tagged `bank_holiday`). Fixing these counts as closing a known gap, not as something the evals found. The bank-holiday share isn't inflated: across a real year, 65% of requests have a deadline that a weekends-only count gets wrong.
- **Test stays held out.** Changes are made against `dev` (42 examples). `test` (21) runs at baseline and at the end only.
- **Every result is reported**, regressions included. A label changes only to fix a mistake, and the change is noted here.
