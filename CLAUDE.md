# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

M0 (scaffold) and M1 (deterministic OpenAlex retrieval) are done. M2 onwards is not started. Implement milestone by milestone (see Milestones) and update this file as decisions are made.

Layout:
- `src/research_agent/config.py`: typed TOML config (`[limits]`, `[openalex]`).
- `src/research_agent/cli.py`: Typer app (`show-config`, `search`).
- `src/research_agent/retrieval/`: no-LLM retrieval. `openalex.py` (pure `build_params`/`parse_work`/`parse_response` plus `OpenAlexClient`), `abstract.py`, `cache.py`, `models.py` (`Work`, `SearchResult`), `errors.py` (all derive from `RetrievalError`).
- `configs/base.toml`, `tests/` (with `tests/fixtures/openalex_search.json`, a trimmed recorded response).

`memory/`, `projects/`, `scripts/` and `secrets/` are empty. Never read, print or commit anything in `secrets/`.

Retrieval facts (verified against the live API on 2026-11-01):
- The API key is optional. Set `OPENALEX_API_KEY` to use one; it is sent as an `Authorization: Bearer` header and never put in URLs or cache keys. Without a key the daily budget is small ($0.10, shared per IP).
- Searches use `search.title_and_abstract`, so every match is on text we can show as evidence.
- Retry only on 429/500/502/503/504 and network errors. A `Retry-After` longer than `max_backoff_seconds` fails fast.
- `SearchResult` carries `total_count`, `missing_abstract_count`, `skipped_records` and `from_cache` so later milestones can report limitations.

## Working principles

Inherited from the parent `git_acorn/CLAUDE.md`:

1. Don't assume.
2. Minimum code that solves the problem.
3. Touch only what you must.
4. Define success criteria.

## Purpose and scope

A research agent that answers questions about GeoAI and remote-sensing literature with concise, referenced technical answers. It searches OpenAlex, reasons over the retrieved evidence, decides whether another search is needed, separates evidence from interpretation, and reports limitations.

**In scope for v1**
- Single-turn question answering from the CLI.
- One LLM agent with tool calling.
- OpenAlex as the only source, using titles, abstracts and metadata.

**Out of scope for v1** (do not add without an explicit request)
- Full-text PDF processing.
- Vector databases or embeddings.
- Agent frameworks (LangChain, LlamaIndex, etc.) and multi-agent designs.
- Memory across questions.
- Web UI or HTTP API.
- Other databases (Semantic Scholar, arXiv, Scopus, ...).

## Confirmed decisions

- LLM access goes through a provider-agnostic `LLMClient` interface, with one concrete adapter first.
- Search source: OpenAlex.
- Interface: CLI.
- Tooling: Python 3.12+, `uv`, `pyproject.toml`, `src` layout, `pytest`, `ruff` (same as `ndvi-forecast`).

## Open decisions

Ask the user before the milestone that needs each one. Do not assume.

- Which provider and model the first adapter targets (needed for M2).
- Deployment platform (not needed before M5 is done).
- Whether to get a free OpenAlex API key (recommended before heavy use; needed for the larger daily budget). The code works without one.
- The evaluation question set and who curates it (needed for M4).

## Architecture

One agent, one loop:

`cli` → `agent` loop → `LLMClient` (adapter per provider) and tools (`search_literature` over OpenAlex)

- The agent loop sends the question and tool definitions to the LLM, runs the tool calls it requests, appends the results, and repeats until the LLM answers or a budget is exhausted.
- Keep pure logic apart from I/O. These parts are pure and unit-tested on their own: reconstructing abstracts from OpenAlex's inverted index, truncation, budget accounting, and answer and citation validation.
- Network, LLM and filesystem access live in thin adapters.
- The agent returns a structured `Answer` containing claims, references and limitations. Each claim is tagged as evidence (with the IDs of the works it cites) or interpretation.
- Budgets are enforced in code, never only in the prompt.

## Resource limits

A `Budget` object is checked on every iteration. These are proposed defaults, all set in the TOML config and never hardcoded elsewhere.

| Limit | Default |
|---|---|
| Agent iterations | 5 |
| Search calls | 3 |
| Results per search | 8 |
| Abstract length | ~1,200 characters |
| Retrieved context sent to the LLM | ~6k tokens |
| Max output tokens per call | ~800 |
| Total tokens per question | ~20k |

- When any limit is hit, the agent is forced to answer from the evidence it already has and must state in the limitations that a limit was reached.
- Cache OpenAlex responses on disk so development and tests do not repeat network calls.
- Report actual token use and search calls for each run.

## Evidence vs interpretation

Behavioural contract for answers:

- Every factual claim cites work IDs that were actually retrieved in this run. Code rejects citations to IDs that were not retrieved.
- Interpretation, synthesis and inference are labelled as such, not presented as findings.
- Limitations are always reported. These include:
  - abstract-only evidence
  - OpenAlex-only coverage
  - works with missing abstracts
  - few or no results
  - a budget being reached
  - possibly outdated evidence
- If the evidence does not support an answer, say so rather than fill the gap from model memory.

## Engineering principles

- Pure core, thin I/O shell.
- Typed configuration (TOML) with typed config objects. No hardcoded limits.
- Deterministic where possible, such as seeds, temperature and cache keys.
- Never silently swallow errors or drop results. Log them and surface them in the answer or exit code.
- Network calls need explicit timeouts, bounded retries and backoff.
- Secrets come from environment variables only. Never log or commit them.
- Use `pathlib` for paths and avoid global mutable state.
- No speculative abstractions and no empty placeholder modules.
- The shell is PowerShell on Windows, so don't write bash-only scripts.

## Testing strategy

- Unit tests cover budgets, truncation, abstract reconstruction, citation validation and config parsing.
- Agent-loop tests use a scripted stub `LLMClient` that returns pre-defined tool calls and answers. They cover:
  - the iteration cap
  - the search cap
  - forced finalisation when a budget is exhausted
  - rejection of citations to unretrieved works
- OpenAlex tests use recorded fixture responses.
- The default `uv run pytest` makes no network calls and costs nothing.
- Live smoke tests are opt-in behind an explicit marker or flag.
- From M4, keep a small hand-written evaluation set (about 10–20 questions) with expected properties, not exact text.

## Commands

```bash
uv sync
uv run pytest
uv run pytest tests/test_config.py::test_missing_file   # single test
uv run ruff check .
uv run research-agent --help
uv run research-agent show-config                       # reads configs/base.toml from the cwd
uv run research-agent search "land cover" --year-from 2022 -n 3 --abstracts
uv run pytest -m live                                   # opt-in: one real OpenAlex call
```

## Milestones

Each milestone is done only when its success criteria are met and verified by running them.

| # | Deliverable | Success criteria |
|---|---|---|
| M0 | Scaffold: `pyproject.toml`, config loading, CLI skeleton | `uv sync`, `uv run pytest` and `uv run ruff check .` pass; CLI `--help` runs |
| M1 | OpenAlex `search_literature` tool with disk cache | Fixture-based tests pass; abstracts reconstructed and truncated correctly; a cache hit makes no network call |
| M2 | `LLMClient` interface, stub client, first real adapter | Stub-based tests pass; the adapter makes one tool-calling round trip (live, opt-in) |
| M3 | Agent loop with budgets and structured `Answer` | Iteration and search caps are enforced in tests; forced finalisation works |
| M4 | Citation validation, limitations reporting, evaluation set | Unretrieved citations are rejected; every answer carries limitations; the evaluation set runs |
| M5 | Hardening: logging, cost reporting, error handling | Failures give clear messages and non-zero exit codes; token and search counts are reported per run |

Priorities, in order: correct citations, budget enforcement, tests, clarity, simplicity.
