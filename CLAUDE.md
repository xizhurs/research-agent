# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

M0 (scaffold), M1 (deterministic OpenAlex retrieval), M2 (LLM interface, stub, OpenAI adapter), M3 (agent loop, budgets, structured answer), M4 (citation validation, limitations, evaluation set), M5 (hardening) and M6 (global extreme rainfall investigation) are done. All planned milestones are complete. Implement milestone by milestone (see Milestones) and update this file as decisions are made.

Layout:
- `src/research_agent/config.py`: typed TOML config (`[limits]`, `[openalex]`, `[llm]`, `[reporting]`, `[rainfall]`).
- `src/research_agent/cli.py`: Typer app (`show-config`, `search`, `rainfall`, `check-credentials`, `ask`, `eval`; global `-v`/`-vv`). The console script calls `cli.run`, which wraps `app`.
- `src/research_agent/logging_setup.py`: log levels, stderr handler, secret redaction.
- `src/research_agent/credentials.py`: loads and validates `OPENALEX_API_KEY` / `OPENAI_API_KEY` (see Credentials).
- `src/research_agent/retrieval/`: no-LLM retrieval. `openalex.py` (pure `build_params`/`parse_work`/`parse_response` plus `OpenAlexClient`), `abstract.py`, `cache.py`, `models.py` (`Work`, `SearchResult`), `errors.py` (all derive from `RetrievalError`).
- `src/research_agent/llm/`: `types.py` (provider-agnostic `LLMClient` protocol, messages, `ToolSpec`/`ToolCall`, `LLMResponse`, `Usage`), `stub.py` (`StubLLMClient`, `text_response`, `tool_call_response`), `openai_client.py` (Responses-API adapter), `errors.py` (all derive from `LLMError`).
- `src/research_agent/agent/`: `loop.py` (`ResearchAgent`), `budget.py` (`Budget`, `estimate_tokens`), `tools.py` (`search_literature` and `analyze_extreme_rainfall` schemas, argument parsing, result formatting), `answer.py` (`Answer`, `Claim`, JSON parsing, citation validation, fallback), `limitations.py` (code-derived limitations), `prompt.py`.
- `src/research_agent/rainfall/`: no-LLM extreme-rainfall analysis. `client.py` (`RainfallClient.analyze`, `parse_archive`, `build_archive_params`), `geocoding.py` (pure parse and disambiguation), `analysis.py` (pure Rx1day statistics and caveats), `models.py` (`Location`, `RainfallAnalysis`, `Provenance`...), `errors.py` (all derive from `RainfallError`).
- `src/research_agent/evaluation.py` and `evals/cases.toml`: evaluation harness and the case set (see Evaluation).
- `configs/base.toml` (adds `[rainfall]` and `limits.max_rainfall_calls`), `tests/` (with `tests/fixtures/openalex_search.json` and `openmeteo_*.json`, trimmed recorded responses).

`memory/`, `projects/`, `scripts/` and `secrets/` are empty. Never read, print or commit anything in `secrets/`.

LLM facts (verified live on 2026-11-01):
- First adapter: OpenAI Responses API, model `gpt-6-luna`, `reasoning_effort = "none"` (both in `configs/base.toml`). Chat Completions is not used: some current models only support function calling on Responses.
- The agent loop must use only `LLMClient` and the types in `llm/types.py`. Pass `LLMResponse.to_message()` back unchanged: its `provider_state` holds the model's output items, which the adapter replays verbatim.
- `allow_tools=False` sets `tool_choice="none"`; use it to force a final answer when a budget is exhausted.
- No `temperature` is sent (reasoning-capable models reject it). Retries (429/5xx/connection) are done by the OpenAI SDK, configured by `llm.max_retries`; the adapter adds none.
- Reasoning tokens count against `max_output_tokens`, so keep reasoning at `none`/low or answers can be cut off (`stop_reason == "length"`).
- Auth errors deliberately omit the server text (it can contain a masked key fragment). Invalid tool-call JSON does not raise: the adapter returns a `ToolCall` with empty `arguments` and an `error` string, and the loop sends that error back to the model (no search charged).
- Use `StubLLMClient` for all agent-loop tests; it raises `AssertionError` if called more often than scripted.
- The OpenAI SDK (3.x) uses `httpx2`, not `httpx`; mock the SDK object (`responses.create`), not the transport.

Agent facts (M3):
- `ResearchAgent(llm, search, limits).run(question)` returns `AgentResult(answer, stats)`. It depends only on the `LLMClient` protocol and a `LiteratureSearch` (anything with `OpenAlexClient.search`'s signature), so tests use `StubLLMClient` and a fake search.
- Caps enforced in code: at most `max_iterations` model calls; the last allowed call (or any call once searches or tokens run out) is made with `allow_tools=False`; searches beyond `max_search_calls` are refused (every tool call ID still gets a result); retrieved text is cut to `max_context_tokens` (estimated at 4 chars per token) in relevance order, and works the model never saw are not citable. Every limit hit adds a sentence to `Answer.limitations`.
- The token cap is predictive: a tool turn runs only if there is room for it and for a final call of similar size. The final call is always made, so actual usage can exceed `max_total_tokens`; that is reported as a limitation.
- The final answer is JSON text (`claims` with `kind` evidence/interpretation and `sources`, plus `limitations`). If the model ignores the format, `build_answer` returns an unstructured interpretation with an explicit limitation instead of crashing.
- References are built from retrieved works, never from model text. "Retrieved" means shown to the model in full; a work hidden by the context budget is not citable.
- Failures: `InvalidQueryError`, bad tool arguments and malformed tool-call JSON go back to the model and cost no search; other `RetrievalError`s cost a search and add a limitation; `AuthError` and `LLMError` abort the run.
- Questions longer than `limits.max_question_chars` (or empty) raise `InvalidQuestionError` before any model call, so an oversized input cannot cause an uncapped first call.
- A cache write failure (disk full, read-only folder, OneDrive lock) is logged and ignored; the cache is only an optimisation.
- `RunStats.cached_searches` counts searches answered from the local cache (no OpenAlex cost); `ask` prints it in its stderr footer.
- Terminal output can contain characters the console cannot encode; the CLI replaces them with `?` instead of crashing.

Citations and limitations (M4):
- `validate_citations` (in `parse_answer`) removes citations to works that were never retrieved and lists them in `Answer.rejected_citations`. An "evidence" claim left with no valid citation (including one that cited nothing) is downgraded to "interpretation". Both cases add a limitation. Claims are never silently dropped.
- Every answer carries limitations. Order: the model's own, then loop notes (budget hits, search failures, truncation), then `evidence_limitations`. That function always includes the OpenAlex/abstract-only caveat and adds, when they apply: no search run, no usable works, thin evidence (`reporting.thin_evidence_works`), only top-ranked results read, year cutoff on all searches, works without abstracts, truncated abstracts, dated evidence (`reporting.stale_after_years`, measured against an injectable clock).
- `Answer.structured` is False for the plain-text fallback. `RunStats.works_shown` counts the distinct works the model saw.
- Known gap: the agent is not restricted to GeoAI. For an off-topic question it still searches OpenAlex (which covers all fields) and may label a statement about missing evidence as "evidence". Decide on a scope guard before relying on it for off-topic input.

Evaluation:
- `uv run research-agent eval [--only ID]` runs `evals/cases.toml` with the real model and OpenAlex (about 78k tokens and 25 searches for the 12 literature cases on 2026-11-01; all 12 passed). The 5 rainfall cases (added 2026-11-02, about 20k tokens) passed on a live run after one prompt fix; on 2026-11-02 the 12 literature cases passed 11/12 once (see Rainfall facts, known gaps). It exits non-zero if any case fails; a provider or search failure fails only that case, while an auth failure aborts the run.
- Cases state expected properties (`min_searches`, `min_references`, `min_rainfall_calls`, `min_data_references`, `keywords_any`, `max_evidence_claims`, `forbidden_phrases`), never exact text. Every case is also checked against universal invariants in `evaluation.check_result` (structured answer, limitations present, every citation (`W…` or `R…`) resolves to a reference, evidence claims cited, budgets respected, token overshoot reported).
- The cases were drafted by the assistant and still need review by the project owner. The off-topic case checks invariants only.
- Offline tests cover the checker with fake results; the full eval is never part of `pytest`.

Rainfall facts (M6; Open-Meteo verified live on 2026-11-02):
- `ResearchAgent(llm, search, limits, reporting, rainfall=...)`: `rainfall` is any object with `RainfallClient.analyze`'s signature (`RainfallAnalyzer`) and is optional; without it the agent is the literature-only agent (original tool list and prompt, unchanged). `ask` always passes the real client.
- Tool `analyze_extreme_rainfall(city, country_code?, region?, start_year?, end_year?, event_date?)`. One call = geocode + archive fetch + compute. Results get IDs `R1`, `R2`... assigned by the loop; they are the numerical citation keys (`Answer.data_references`), like `W…` for works. Only results shown to the model in full are citable.
- Budgets: `limits.max_rainfall_calls` (default 3). Bad arguments and `InvalidRainfallQueryError` cost nothing; ambiguous, not-found, network and insufficient-data outcomes cost a call and add a limitation. A final answer is forced only when BOTH the search and rainfall budgets are spent. Rainfall text counts against the shared `max_context_tokens`; a result that does not fit is refused and not citable.
- Open-Meteo geocoding: `name`, `count`, `countryCode`, `language`, `format=json`; no match = a response with no `results` key; the result carries `id`, `name`, `latitude`, `longitude`, `country_code`, `admin1`, `population` (often missing for small places), `timezone`. `region` is matched locally (exact, case-insensitive).
- Open-Meteo archive: `daily=precipitation_sum` (mm), `timezone=auto` (local calendar days), start from 1940-01-01, `end_date` up to today. Errors are HTTP 400 with `{"error": true, "reason": ...}`. The most recent ~5 days and 1940-01-01 are `null`, so nulls are handled, never filled. A 1950-2025 request is about 0.5 MB and about 1.6 s.
- **Always pin `models=era5`.** Without it the API picks another product and grid cell: for Zhengzhou on 2021-07-20 the unpinned request returned 152 mm at (34.76, 113.58) and the pinned one 79-82 mm on the 0.25 degree grid. Values also change with the grid cell the coordinates snap to, so the grid cell actually used is reported.
- Disambiguation: one candidate, or a top candidate whose population is at least `rainfall.dominance_ratio` times the runner-up's (Paris, France qualifies; Springfield does not), is accepted and the alternatives are listed in the limitations; otherwise `AmbiguousLocationError` lists the candidates and no analysis runs. The tool result tells the model not to guess.
- Statistics: a year needs `min_year_coverage` of its days and must be complete (`year <= last complete year`); fewer than `min_years` valid years withholds ranks, percentiles and the baseline comparison; ties report the earliest date; no extreme-value fitting, no trends.
- Cache: raw JSON in `.cache/rainfall/` (`rainfall.cache_ttl_days`), same `ResponseCache` as OpenAlex. No credentials are used or cached.
- Open-Meteo free tier: non-commercial use, CC BY 4.0 attribution, 10,000 calls/day (from their pricing page; not re-verified). Commercial use needs a paid plan. 429 handling is generic (Retry-After or backoff) and has not been observed live.
- Known gaps: numbers inside a claim are not machine-checked against the cited analysis (only that the cited `R…` exists); ERA5 at 0.25 degrees smooths local extremes, so rankings for flash events can look mild; one literature case (`change-detection`) returned an unstructured answer once in four live runs after the two-tool prompt was introduced (3 of 3 reruns passed; not established whether the prompt matters).

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
- OpenAlex as the only literature source, using titles, abstracts and metadata. (M6 adds Open-Meteo/ERA5 as a numerical source for the rainfall tool, by explicit request.)

**Out of scope for v1** (do not add without an explicit request)
- Full-text PDF processing.
- Vector databases or embeddings.
- Agent frameworks (LangChain, LlamaIndex, etc.) and multi-agent designs.
- Memory across questions.
- Web UI or HTTP API.
- Other databases (Semantic Scholar, arXiv, Scopus, ...).

## Confirmed decisions

- LLM access goes through a provider-agnostic `LLMClient` interface, with one concrete adapter first.
- Search source: OpenAlex. Rainfall source (M6): Open-Meteo geocoding + archive API, ERA5 reanalysis, annual maximum daily precipitation (Rx1day).
- Interface: CLI.
- Tooling: Python 3.12+, `uv`, `pyproject.toml`, `src` layout, `pytest`, `ruff` (same as `ndvi-forecast`).

## Open decisions

Ask the user before the milestone that needs each one. Do not assume.

- Deployment platform (nothing is built for it; decide before packaging or scheduling runs).
- Whether to get a free OpenAlex API key (recommended before heavy use; needed for the larger daily budget). The code works without one.
- Who reviews and extends `evals/cases.toml` (a draft exists), and whether to add a GeoAI scope guard for off-topic questions.

## Logging, errors and exit codes (M5)

- Logs go to stderr; answers go to stdout. Default level is WARNING. `research-agent -v ...` logs each model turn, token counts and each search; `-vv` adds debug details. The flag goes before the command: `research-agent -v ask "..."`.
- Only `research_agent.*` loggers follow the flag. `httpx`, `openai` and friends stay at WARNING because their debug output can contain full prompts. The question text is never logged.
- `REDACTOR` (`logging_setup.py`) replaces any loaded API key with `<redacted>` in log messages and tracebacks. It is fed by `_credentials_or_exit`; any new code path that loads credentials must go through that function.
- Typer is configured with `pretty_exceptions_enable=False` and `pretty_exceptions_show_locals=False`: Typer's default crash output prints local variables, which can include API keys (verified). Do not turn this back on.
- Exit codes: 0 success; 1 any failure (expected failures print one `error: ...` line; unexpected exceptions print one redacted `error: unexpected failure (...)` line, with a traceback only under `-vv`); 2 command-line usage error (Click). `eval` exits 1 if any case fails.

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
- CI (`.github/workflows/ci.yml`) runs on pull requests to `main`: `uv sync --locked`, `ruff check`, `pytest`, on Ubuntu and Windows with Python 3.12 and 3.13. It has no secrets and must stay offline: never add the `live` marker, `eval` or API keys to it. Actions are pinned by commit SHA with the version in a comment; update them deliberately. The `CI passed` job is the single check to require in branch protection. Keep `uv.lock` committed and current, or `--locked` fails the job.
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
uv run research-agent -v ask "..."                      # progress log on stderr
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
| M5 (done) | Hardening: logging, cost reporting, error handling | Failures give clear messages and non-zero exit codes; token and search counts are reported per run |
| M6 (done) | Global extreme-rainfall tool (geocoding + ERA5 Rx1day) in the agent loop, `R…` citations, rainfall budget, `rainfall` CLI command, 5 eval cases | Offline tests pass; `rainfall` command prints statistics and provenance and is cached on the second run; ambiguous city yields no analysis; fabricated `R…` IDs are rejected; caps hold; the new eval cases pass live |

Priorities, in order: correct citations, budget enforcement, tests, clarity, simplicity.
