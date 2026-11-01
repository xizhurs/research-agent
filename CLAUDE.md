# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

M0 (scaffold), M1 (deterministic OpenAlex retrieval), M2 (LLM interface, stub, OpenAI adapter) M3 (agent loop, budgets, structured answer) and M4 (citation validation, limitations, evaluation set) are done. M5 is not started. Implement milestone by milestone (see Milestones) and update this file as decisions are made.

Layout:
- `src/research_agent/config.py`: typed TOML config (`[limits]`, `[openalex]`, `[llm]`, `[reporting]`).
- `src/research_agent/cli.py`: Typer app (`show-config`, `search`, `check-credentials`, `ask`, `eval`).
- `src/research_agent/credentials.py`: loads and validates `OPENALEX_API_KEY` / `OPENAI_API_KEY` (see Credentials).
- `src/research_agent/retrieval/`: no-LLM retrieval. `openalex.py` (pure `build_params`/`parse_work`/`parse_response` plus `OpenAlexClient`), `abstract.py`, `cache.py`, `models.py` (`Work`, `SearchResult`), `errors.py` (all derive from `RetrievalError`).
- `src/research_agent/llm/`: `types.py` (provider-agnostic `LLMClient` protocol, messages, `ToolSpec`/`ToolCall`, `LLMResponse`, `Usage`), `stub.py` (`StubLLMClient`, `text_response`, `tool_call_response`), `openai_client.py` (Responses-API adapter), `errors.py` (all derive from `LLMError`).
- `src/research_agent/agent/`: `loop.py` (`ResearchAgent`), `budget.py` (`Budget`, `estimate_tokens`), `tools.py` (`search_literature` schema, argument parsing, result formatting), `answer.py` (`Answer`, `Claim`, JSON parsing, citation validation, fallback), `limitations.py` (code-derived limitations), `prompt.py`.
- `src/research_agent/evaluation.py` and `evals/cases.toml`: evaluation harness and the case set (see Evaluation).
- `configs/base.toml`, `tests/` (with `tests/fixtures/openalex_search.json`, a trimmed recorded response).

`memory/`, `projects/`, `scripts/` and `secrets/` are empty. Never read, print or commit anything in `secrets/`.

LLM facts (verified live on 2026-11-01):
- First adapter: OpenAI Responses API, model `gpt-6-luna`, `reasoning_effort = "none"` (both in `configs/base.toml`). Chat Completions is not used: some current models only support function calling on Responses.
- The agent loop must use only `LLMClient` and the types in `llm/types.py`. Pass `LLMResponse.to_message()` back unchanged: its `provider_state` holds the model's output items, which the adapter replays verbatim.
- `allow_tools=False` sets `tool_choice="none"`; use it to force a final answer when a budget is exhausted.
- No `temperature` is sent (reasoning-capable models reject it). Retries (429/5xx/connection) are done by the OpenAI SDK, configured by `llm.max_retries`; the adapter adds none.
- Reasoning tokens count against `max_output_tokens`, so keep reasoning at `none`/low or answers can be cut off (`stop_reason == "length"`).
- Auth errors deliberately omit the server text (it can contain a masked key fragment). Invalid tool-call JSON raises `LLMResponseError`; the M3 loop decides how to handle it.
- Use `StubLLMClient` for all agent-loop tests; it raises `AssertionError` if called more often than scripted.
- The OpenAI SDK (3.x) uses `httpx2`, not `httpx`; mock the SDK object (`responses.create`), not the transport.

Agent facts (M3):
- `ResearchAgent(llm, search, limits).run(question)` returns `AgentResult(answer, stats)`. It depends only on the `LLMClient` protocol and a `LiteratureSearch` (anything with `OpenAlexClient.search`'s signature), so tests use `StubLLMClient` and a fake search.
- Caps enforced in code: at most `max_iterations` model calls; the last allowed call (or any call once searches or tokens run out) is made with `allow_tools=False`; searches beyond `max_search_calls` are refused (every tool call ID still gets a result); retrieved text is cut to `max_context_tokens` (estimated at 4 chars per token) in relevance order, and works the model never saw are not citable. Every limit hit adds a sentence to `Answer.limitations`.
- The token cap is predictive: a tool turn runs only if there is room for it and for a final call of similar size. The final call is always made, so actual usage can exceed `max_total_tokens`; that is reported as a limitation.
- The final answer is JSON text (`claims` with `kind` evidence/interpretation and `sources`, plus `limitations`). If the model ignores the format, `build_answer` returns an unstructured interpretation with an explicit limitation instead of crashing.
- References are built from retrieved works, never from model text. "Retrieved" means shown to the model in full; a work hidden by the context budget is not citable.
- Failures: `InvalidQueryError` and bad tool arguments go back to the model and cost no search; other `RetrievalError`s cost a search and add a limitation; `AuthError` and `LLMError` abort the run.
- Terminal output can contain characters the console cannot encode; the CLI replaces them with `?` instead of crashing.

Citations and limitations (M4):
- `validate_citations` (in `parse_answer`) removes citations to works that were never retrieved and lists them in `Answer.rejected_citations`. An "evidence" claim left with no valid citation (including one that cited nothing) is downgraded to "interpretation". Both cases add a limitation. Claims are never silently dropped.
- Every answer carries limitations. Order: the model's own, then loop notes (budget hits, search failures, truncation), then `evidence_limitations`. That function always includes the OpenAlex/abstract-only caveat and adds, when they apply: no search run, no usable works, thin evidence (`reporting.thin_evidence_works`), only top-ranked results read, year cutoff on all searches, works without abstracts, truncated abstracts, dated evidence (`reporting.stale_after_years`, measured against an injectable clock).
- `Answer.structured` is False for the plain-text fallback. `RunStats.works_shown` counts the distinct works the model saw.
- Known gap: the agent is not restricted to GeoAI. For an off-topic question it still searches OpenAlex (which covers all fields) and may label a statement about missing evidence as "evidence". Decide on a scope guard before relying on it for off-topic input.

Evaluation:
- `uv run research-agent eval [--only ID]` runs `evals/cases.toml` with the real model and OpenAlex (about 78k tokens and 25 searches for 12 cases on 2026-11-01; all 12 passed). It exits non-zero if any case fails; a provider or search failure fails only that case, while an auth failure aborts the run.
- Cases state expected properties (`min_searches`, `min_references`, `keywords_any`, `max_evidence_claims`, `forbidden_phrases`), never exact text. Every case is also checked against universal invariants in `evaluation.check_result` (structured answer, limitations present, every citation resolves to a reference, evidence claims cited, budgets respected, token overshoot reported).
- The cases were drafted by the assistant and still need review by the project owner. The off-topic case checks invariants only.
- Offline tests cover the checker with fake results; the full eval is never part of `pytest`.

Retrieval facts (verified against the live API on 2026-11-01):
- The OpenAlex key is optional (small anonymous budget without it: $0.10/day shared per IP). It is sent as an `Authorization: Bearer` header and never put in URLs or cache keys; the client also redacts it from server error text.
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

- Deployment platform (not needed before M5 is done).
- Whether to get a free OpenAlex API key (recommended before heavy use; needed for the larger daily budget). The code works without one.
- Who reviews and extends `evals/cases.toml` (a draft exists), and whether to add a GeoAI scope guard for off-topic questions.

## Credentials

- Keys come from the process environment first, then a local `.env` (copy `.env.example`; `.env` is git-ignored, `.env.example` is tracked and holds placeholders only). Use `--env-file` or `RESEARCH_AGENT_ENV_FILE` to point at a file elsewhere.
- `load_credentials` never mutates `os.environ`. Keys are held in `Secret`, whose `repr`/`str`/format are redacted; call `.reveal()` only at the point of use. Errors name the variable, never the value.
- Never read, print or commit `.env`. To check configuration use `uv run research-agent check-credentials` (offline: presence and format only; it cannot tell if a provider accepts the key).
- `OPENAI_API_KEY` is used by `OpenAIClient.from_credentials`, which raises `CredentialError` if it is missing.

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
uv run research-agent ask "Which deep learning methods are used for crop type mapping?"
uv run research-agent check-credentials                 # never prints key values
uv run research-agent eval                              # real model + OpenAlex; costs tokens
uv run pytest -m live                                   # opt-in: real OpenAlex search + OpenAI tool-calling round trip
```

## Milestones

Each milestone is done only when its success criteria are met and verified by running them.

| # | Deliverable | Success criteria |
|---|---|---|
| M0 | Scaffold: `pyproject.toml`, config loading, CLI skeleton | `uv sync`, `uv run pytest` and `uv run ruff check .` pass; CLI `--help` runs |
| M1 | OpenAlex `search_literature` tool with disk cache | Fixture-based tests pass; abstracts reconstructed and truncated correctly; a cache hit makes no network call |
| M2 (done) | `LLMClient` interface, stub client, first real adapter | Stub-based tests pass; the adapter makes one tool-calling round trip (live, opt-in) |
| M3 (done) | Agent loop with budgets and structured `Answer` | Iteration and search caps are enforced in tests; forced finalisation works |
| M4 (done) | Citation validation, limitations reporting, evaluation set | Unretrieved citations are rejected; every answer carries limitations; the evaluation set runs |
| M5 | Hardening: logging, cost reporting, error handling | Failures give clear messages and non-zero exit codes; token and search counts are reported per run |

Priorities, in order: correct citations, budget enforcement, tests, clarity, simplicity.
