# AGENTS.md

A triage agent for Freedom of Information requests to a UK council, built as a LangGraph `StateGraph`. The graph and its nodes are described in [README.md](README.md).

## Design rules

- Build with plain LangGraph pieces: `StateGraph`, nodes, edges and `interrupt`.
- Judgement nodes call the LLM and return structured output. Routing and the escalate-or-send rule are deterministic code.
- Regime classification runs first. Non-FOI requests leave the graph before any FOI work.
- The agent sends on its own only in low-risk cases. When in doubt it escalates to an FOI officer.

## Working here

- Set up: `python -m venv .venv`, then `.venv/Scripts/python -m pip install -e ".[dev]"`.
- On this Windows machine the venv's `.exe` launchers are blocked, so run CLIs through Python, e.g. `.venv/Scripts/python -c "from langgraph_cli.cli import cli; cli()" dev`.
- Deterministic nodes (deadline, routing, escalation rule) get unit tests.
- Finish with `pytest` and `ruff check .` passing.
- Leave changes uncommitted. The human writes every commit, because reviewers read the history.
- Docs describe only what exists in the repo. Write in British English.
