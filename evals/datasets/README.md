# Golden dataset

[golden.jsonl](golden.jsonl) holds 51 FOI requests to the fictional Northbrook Borough Council, each labelled with what an FOI officer would want the agent to do. The labels describe the ideal outcome, not what the current version can do, so some examples fail on purpose until the agent improves.

[tests/test_golden_dataset.py](../../tests/test_golden_dataset.py) checks every example's format, that every labelled id exists in the knowledge base, and that the labels don't contradict each other.

## Format

One JSON object per line:

```json
{"inputs": {"request": "...", "received": "2026-11-02"},
 "outputs": {"regime": "FOI", "action": "send", "published_id": "PS-003",
             "exemptions": ["s21"], "relevant_ids": ["PS-003", "DL-2026-016"], "deadline": "2026-12-01"},
 "metadata": {"slice": "vocab_mismatch", "category": "edge", "split": "dev",
              "why": "...", "tags": ["bank_holiday"], "source": "new", "review": "pending"}}
```

An output field is left out when it doesn't apply to the example, and its evaluator skips that example.

| Output | Meaning |
| --- | --- |
| `regime` | `FOI`, `EIR`, `SAR` or `BAU`. Left out for requests that genuinely mix regimes |
| `action` | `send` (a signpost the agent may send itself), `route` (SAR or business as usual), `escalate` (to an FOI officer) or `clarify` (ask the requester to narrow or explain the request) |
| `published_id` | The record that fully answers the request, or `null` if nothing published does. A record that answers only part of the request, or the wrong period, doesn't count |
| `exemptions` | FOIA sections an officer would expect to consider, written like `s40`. Includes `s21` for signposts |
| `relevant_ids` | Records a good search should find: the published record, precedents in the disclosure log and the guidance that applies |
| `deadline` | The statutory deadline for FOI and EIR requests |

## Deadlines

Deadlines follow FOIA section 10. Day one is the first working day after receipt, and the deadline is the 20th working day. Section 10(6) excludes bank holidays in any part of the UK, so St Andrew's Day (30 November) and Scotland's 2 January holiday count, as well as Christmas and New Year. 25 examples have deadlines that a weekends-only count gets wrong, tagged `bank_holiday`.

## Coverage

The brief asks for normal cases, edge cases, and failures that would matter in the real world. Each example has a `category` for one of those three, and a finer `slice`:

| Category | Slices | Examples |
| --- | --- | --- |
| `normal` | `published_exact`, `plain_provide`, `routine_sar`, `routine_bau`, `routine_eir` | 14 |
| `edge` | `vocab_mismatch`, `multi_part`, `wrong_period`, `hidden_exemption`, `precedent`, `too_broad`, `deadline` | 19 |
| `high_risk` | `regime_trap`, `mixed`, `adversarial`, `unsafe_send_bait` | 18 |

High-risk examples are the ones where a wrong answer can't be undone once it is sent. They include subject access requests that cite FOI, requests that mix regimes, prompt injection, and requests designed to tempt an automatic send: half-published requests, the wrong year, and a refusal presented as already published.

## Splits

34 examples are in `dev` and 17 in `test`, assigned across slices so that both splits cover every category. Iterate on `dev` only. Run `test` once at baseline and once at the end.

## How the labels were made

Claude drafted the requests and labels from the knowledge base and ICO guidance, without running the agent on them, so the labels don't simply agree with the agent's own output. 13 examples come from [examples/requests.jsonl](../../examples/requests.jsonl), recorded in `source`. Each example has a `why` explaining its label. `review` is `pending` until a person has checked the label.
