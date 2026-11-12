# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

M0 (scaffold), M1 (deterministic OpenAlex retrieval), M2 (LLM interface, stub, OpenAI adapter), M3 (agent loop, budgets, structured answer), M4 (citation validation, limitations, evaluation set), M5 (hardening), M6 (global extreme rainfall investigation), M7 (extreme dry analysis), M8 (El Nino / La Nina events), M9 (optional Gradio web interface) and M10 (follow-up questions, added 2026-11-11) are done. All planned milestones are complete. Implement milestone by milestone (see Milestones) and update this file as decisions are made.

Layout:
- `src/research_agent/config.py`: typed TOML config (`[limits]`, `[openalex]`, `[llm]`, `[reporting]`, `[rainfall]`, `[enso]`, `[conversation]`).
- `src/research_agent/cli.py`: Typer app (`show-config`, `search`, `rainfall`, `enso`, `ui`, `check-credentials`, `ask`, `eval`; global `-v`/`-vv`). The console script calls `cli.run`, which wraps `app`.
- `src/research_agent/logging_setup.py`: log levels, stderr handler, secret redaction.
- `src/research_agent/credentials.py`: loads and validates `OPENALEX_API_KEY` / `OPENAI_API_KEY` (see Credentials).
- `src/research_agent/retrieval/`: no-LLM retrieval. `openalex.py` (pure `build_params`/`parse_work`/`parse_response` plus `OpenAlexClient`), `abstract.py`, `cache.py`, `models.py` (`Work`, `SearchResult`), `errors.py` (all derive from `RetrievalError`).
- `src/research_agent/llm/`: `types.py` (provider-agnostic `LLMClient` protocol, messages, `ToolSpec`/`ToolCall`, `LLMResponse`, `Usage`), `stub.py` (`StubLLMClient`, `text_response`, `tool_call_response`), `openai_client.py` (Responses-API adapter), `errors.py` (all derive from `LLMError`).
- `src/research_agent/agent/`: `loop.py` (`ResearchAgent`), `budget.py` (`Budget`, `estimate_tokens`), `tools.py` (`search_literature`, `analyze_rainfall_extremes` and `analyze_enso_events` schemas, argument parsing, result formatting), `answer.py` (`Answer`, `Claim`, JSON parsing, citation validation, fallback), `limitations.py` (code-derived limitations), `prompt.py`.
- `src/research_agent/rainfall/`: no-LLM extreme-rainfall analysis. `client.py` (`RainfallClient.analyze` returns `(record, Archive)`, `fetch_archive`, `parse_archive`, `build_archive_params`), `climatology.py` (pure baselines for the plots: smoothed day-of-year mean, monthly totals and their percentile band), `geocoding.py` (pure parse and disambiguation), `analysis.py` (pure Rx1day statistics, shared year-validity and caveat helpers), `dryness.py` (pure CDD and annual-total statistics), `models.py` (`Location`, `RainfallAnalysis`, `Provenance`...), `errors.py` (all derive from `RainfallError`).
- `src/research_agent/enso/`: no-LLM El Nino / La Nina event detection on NOAA CPC's RONI file. `analysis.py` (pure: strict `parse_roni`, `find` runs, strength, caveats), `client.py` (`EnsoClient.analyze`, cached download with bounded retries), `models.py` (`EnsoRun`, `EnsoFocus`, `EnsoAnalysis`), `errors.py` (all derive from `EnsoError`).
- `src/research_agent/ui/`: optional Gradio interface, installed with `uv sync --extra ui` (see UI facts). `handlers.py` (plain functions, no Gradio import), `plots.py` (plotly figure, imported lazily), `app.py` (layout only, the only module that imports Gradio).
- `src/research_agent/tracing.py`: optional OpenTelemetry tracing to Langfuse or any OTLP backend (see Tracing).
- `src/research_agent/conversation/`: follow-up questions (see Conversation facts). `models.py` (`Session`, `TurnRecord`, `TaskState`, `EvidenceItem`), `state.py` (pure `apply_turn`), `context.py` (pure `build_context`), `store.py` (`SessionStore` protocol, `InMemorySessionStore`), `service.py` (`ConversationService`). The agent side is `agent/events.py` (`ToolEvent`, `TurnContext`).
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
- `uv run research-agent eval [--only ID]` runs `evals/cases.toml` with the real model and OpenAlex (about 78k tokens and 25 searches for the 12 literature cases on 2026-11-01; all 12 passed). The 5 rainfall cases (added 2026-11-02, about 20k tokens) passed on a live run after one prompt fix; on 2026-11-02 the 12 literature cases passed 11/12 after M6 and 12/12 after M7 (see Rainfall facts, known gaps). The 4 dry cases and 5 rainfall cases passed 9/9 after M7. It exits non-zero if any case fails; a provider or search failure fails only that case, while an auth failure aborts the run.
- Cases state expected properties (`min_searches`, `min_references`, `min_rainfall_calls` (rainfall and ENSO analyses together), `min_data_references`, `expect_metric`, `expect_tool`, `forbid_tool` (both judged on the tool trace), `keywords_any`, `max_evidence_claims`, `forbidden_phrases`), never exact text. Every case is also checked against universal invariants in `evaluation.check_result` (structured answer, limitations present, every citation (`W…` or `R…`) resolves to a reference, evidence claims cited, budgets respected, token overshoot reported).
- The cases were drafted by the assistant and still need review by the project owner. The off-topic case checks invariants only.
- Offline tests cover the checker with fake results; the full eval is never part of `pytest`.

Rainfall facts (M6 and M7; Open-Meteo verified live on 2026-11-02):
- `ResearchAgent(llm, search, limits, reporting, rainfall=...)`: `rainfall` is any object with `RainfallClient.analyze`'s signature (`RainfallAnalyzer`) and is optional; without it the agent is the literature-only agent (original tool list and prompt, unchanged). `ask` always passes the real client.
- Tool `analyze_rainfall_extremes(city, country_code?, region?, start_year?, end_year?, event_date?, metric?, event_year?)` (renamed from `analyze_extreme_rainfall` in M7). `metric` is `wet` (default, Rx1day) or `dry`. One call = geocode + archive fetch + compute. Results get IDs `R1`, `R2`... assigned by the loop; they are the numerical citation keys (`Answer.data_references`), like `W…` for works. Only results shown to the model in full are citable.
- Budgets: `limits.max_rainfall_calls` (default 3). Bad arguments and `InvalidRainfallQueryError` cost nothing; ambiguous, not-found, network and insufficient-data outcomes cost a call and add a limitation. A final answer is forced only when BOTH the search and rainfall budgets are spent. Rainfall text counts against the shared `max_context_tokens`; a result that does not fit is refused and not citable.
- Open-Meteo geocoding: `name`, `count`, `countryCode`, `language`, `format=json`; no match = a response with no `results` key; the result carries `id`, `name`, `latitude`, `longitude`, `country_code`, `admin1`, `population` (often missing for small places), `timezone`. `region` is matched locally (exact, case-insensitive).
- Open-Meteo archive: `daily=precipitation_sum` (mm), `timezone=auto` (local calendar days), start from 1940-01-01, `end_date` up to today. Errors are HTTP 400 with `{"error": true, "reason": ...}`. The most recent ~5 days and 1940-01-01 are `null`, so nulls are handled, never filled. A 1950-2025 request is about 0.5 MB and about 1.6 s.
- **Always pin `models=era5`.** Without it the API picks another product and grid cell: for Zhengzhou on 2021-07-20 the unpinned request returned 152 mm at (34.76, 113.58) and the pinned one 79-82 mm on the 0.25 degree grid. Values also change with the grid cell the coordinates snap to, so the grid cell actually used is reported.
- Disambiguation: one candidate, or a top candidate whose population is at least `rainfall.dominance_ratio` times the runner-up's (Paris, France qualifies; Springfield does not), is accepted and the alternatives are listed in the limitations; otherwise `AmbiguousLocationError` lists the candidates and no analysis runs. The tool result tells the model not to guess.
- Statistics: a year needs `min_year_coverage` of its days and must be complete (`year <= last complete year`); fewer than `min_years` valid years withholds ranks, percentiles and the baseline comparison; ties report the earliest date; no extreme-value fitting, no trends.
- Cache: raw JSON in `.cache/rainfall/` (`rainfall.cache_ttl_days`), same `ResponseCache` as OpenAlex. No credentials are used or cached.
- Open-Meteo free tier: non-commercial use, CC BY 4.0 attribution, 10,000 calls/day (from their pricing page; not re-verified). Commercial use needs a paid plan. 429 handling is generic (Retry-After or backoff) and has not been observed live.
- Dry metric (M7, `rainfall/dryness.py`): a day is dry if it has less than `rainfall.dry_day_mm` (1.0, the ETCCDI CDD convention; exactly 1.0 mm is wet). A missing day breaks a run and is never counted as dry. Yearly CDD restarts on 1 January (ETCCDI sources I found disagree on boundary handling, so this is our own explicit rule, not claimed to be ETCCDI-compliant); the unbroken-spell list is not split at New Year and only uses valid years. Annual total is the sum of all days, deliberately not called PRCPTOT (which counts wet days only). Rank 1 = driest (totals) or longest (spells). `event_year` (dry only) gives a year's total, rank, percent of years wetter and CDD; `event_date` finds the unbroken spell containing the day; an excluded or incomplete year is reported, not ranked.
- Dry and wet share one archive request and cache entry, so a dry analysis after a wet one of the same city makes no network call; it still costs one rainfall call of the budget. `tool_trace` names rainfall calls with their metric (`analyze_rainfall_extremes[dry]`), and eval cases can state `expect_metric` ("wet" or "dry"), judged on that trace.
- Dry caveats (code-added): ERA5 is not a gauge; regional studies (Spain, Vietnam; no global figure found) show ERA5 has too many light-rain days, which can shorten dry spells; a dry spell is not a drought index; yearly CDD restarts on 1 January and mostly reflects dry-season length where there is one. The prompt forbids calling a dry spell or dry year a "drought" unless a retrieved work says so, and tells the model to leave `start_year`/`end_year` unset unless the question names a period (without that rule it chose 1940-2022 for "how dry was 2022", so ranks were not comparable between questions).
- Live plausibility check on 2026-11-02 (1950-2025): baseline mean yearly CDD about 274 days for Cairo, 51 Perth, 50 Madrid; Cairo's longest unbroken spell was 700 days (more than a calendar-year CDD can show); Madrid's driest years by total were 1994, 2005, 2015.
- Known gaps: numbers inside a claim are not machine-checked against the cited analysis (only that the cited `R…` exists); ERA5 at 0.25 degrees smooths local extremes, so rankings for flash events can look mild; place names the geocoder lists twice were reported as ambiguous when the two populations were within `dominance_ratio` (for example "Singapore": 6.1M and 5.6M entries, both Singapore, Singapore); the `same_place_km` merge (below) should resolve this if the two entries are within 25 km, not re-checked; `change-detection` returned an unstructured answer once in four live runs after the two-tool prompt was introduced (M6); 3 of 3 reruns and the full 12-case re-run after M7 (12/12) passed, so it looks like run-to-run variance, but that is not proven.

ENSO facts (M8; NOAA CPC files fetched 2026-11-02):
- Tool `analyze_enso_events(phase?, start_year?, end_year?, year?)`, offered only together with the rainfall tool (`ResearchAgent(..., rainfall=..., enso=...)`; `enso` without `rainfall` raises `ValueError`). Without `enso` the literature-only and two-tool prompts are byte-identical to before (tested). `ask` always passes the real client. The prompt tells the model: events/dates/strength/counts only from the tool, literature for causes, impacts and what a paper counted, both when a question needs both.
- Source: `https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt` (RONI, Relative Oceanic Nino Index): header `SEAS   YR  ANOM`, rows like `DJF  1950 -1.19`, 12 overlapping 3-month seasons per year (DJF...NDJ), 2 decimals, 1950 DJF to JAS 2026 on 2026-11-02, no comment, version or licence lines. NOAA's pages say RONI is the official ENSO index (NWS PIS 26-05, effective Feb 2026; the legacy ONI file `oni.ascii.txt` is still published and has an extra TOTAL column); RONI uses the fixed 1991-2020 base. The file does not say which SST version it uses (a web search said ERSST.v5, NOAA's ONI page says v6, unresolved), so output never claims one. No licence statement found.
- Event rule (NOAA's RONI page): index at or beyond +/-0.5 for at least 5 consecutive overlapping seasons. The page says "greater than 0.5", an older climate.gov page "+0.5 or higher"; we use inclusive (`>=`), which only matters for a value of exactly 0.50. Config: `[enso] threshold`, `min_seasons`.
- Strength labels (weak from 0.5, moderate from 1.0, strong from 1.5, very strong from 2.0, by PEAK |value|) are a convention the user chose to add, not on NOAA's pages. Golden Gate Weather's RONI page (ggweather.com/enso/roni.htm) uses the same class boundaries but requires a class to be reached for 3 consecutive seasons; that stricter rule would lower 12 of 44 labels (for example 1991-92 El Nino very strong -> strong, 2009-10 strong -> moderate). The result's caveat says so. Open decision for the owner: keep peak-based labels or adopt the stricter rule.
- Strict parsing (`parse_roni`): header must be exactly `SEAS YR ANOM`; unknown season code, bad number, non-finite or |value| > 10 (a -99.9 style sentinel), and any missing, duplicate or out-of-order season (a gap would silently split an event) raise `EnsoApiError`; a file that does not parse is never cached. Events are always detected on the whole record and then selected by the year they START in, so a window never cuts an event short. Ties report the earliest peak season.
- Record edges are flagged, never hidden: a run reaching the last season is `ongoing`; a run shorter than 5 seasons at the end is reported as "not yet an event" (`pending`; on 2026-11-02 El Nino-level conditions for MJJ-JAS 2026, 3 seasons); a run starting at the first row (La Nina DJF 1950) is flagged as possibly cut. All three also add a code caveat.
- Budget and citations: ENSO calls share `limits.max_rainfall_calls` (3) and the `R1`, `R2`... namespace with rainfall analyses (`Answer.data_references` holds both; `DataRecord` in `agent/answer.py`). `RunStats.rainfall_calls` therefore counts both, and the `ask` footer says "rainfall/ENSO analysis(es)". Bad arguments and `InvalidEnsoQueryError` cost nothing; network, API-shape and out-of-record outcomes cost a call and add a limitation. A year or period outside the record is only known after the download, so it costs a call.
- Cache: raw text as JSON in `.cache/enso/` (`enso.cache_ttl_days` = 7, the file is updated monthly). No credentials.
- Size: the full-record result is about 1.8k estimated tokens (44 events plus caveats); one live question used about 5.8k tokens in total.
- Verification: the 44 events (22 El Nino, 22 La Nina, 2026-11-02) were re-derived by a separate naive implementation sharing no code and are identical (same start and end seasons); well-known events appear (1982-83, 1997-98 and 2015-16 very strong El Nino; 2010-11 and 2020-23 La Nina). Not verified against NOAA's own published event list: the ggweather page could only be read as flattened text, so its per-year labels could not be matched to events (its totals, El Nino 28 and La Nina 26, differ from ours, 22 and 22, probably because it counts July-June years; not resolved).
- Live eval on 2026-11-02 after M8: all 26 cases passed (193,683 tokens, 30 searches, 14 rainfall/ENSO analyses): the 12 literature cases, the 9 rainfall/dry cases (one hit an Open-Meteo HTTP 503 after retries, was reported as a failed analysis, and the model recovered) and the 5 ENSO cases. The model chose `enso` for the three events/year/current questions, searched only for the literature-only question (3 searches, no ENSO call), and used both tools for the La Nina plus impacts question. One run per case: variance is not measured.
- Known gaps: RONI is not the index of most older papers (ONI), so literature event lists can differ; caveats about the record start and a pending run are shown even when the question is about an unrelated year; the `enso-current-conditions` eval case is time-dependent.

- Time scales (added 2026-11-11): the rainfall tool takes a required `scale` (`yearly`, `monthly`, `daily`; a call without it is treated as daily) and `event_month`. `daily` is the M6/M7 behaviour (`metric` wet or dry). `yearly` (`rainfall/totals.py`, `YearlyAnalysis`): every valid year's total (all listed for periods up to 30 years, with the ratio to the baseline mean), wettest and driest years, and `event_year` ranked with its position against the baseline band. `monthly` (`MonthlyAnalysis`): a month's total (`event_year` + `event_month`), or each month of `event_year`, against the same calendar month in the period (rank) and in the baseline (mean, median, 10th-90th percentile band); also the wettest months. Its default `end_year` is the current year, so finished months of this year count. For yearly and monthly the daily series starts at `min(start_year, baseline_start_year)`, so a short recent period ("last 10 years") still gets the 1991-2020 comparison; ranks cover the requested period only and are withheld below `min_years` totals. A month counts if it has ended and has `min_year_coverage` of its days (month-based, not year-based). Rejected before any request: an unknown scale, `event_date` with a non-daily scale, `event_month` without scale monthly and `event_year`, `event_year` with daily wet. The prompt sends general questions ("how was the rainfall in the last 10 years") to `yearly`; `tool_trace` names the call `[yearly]`, `[monthly]`, `[wet]` or `[dry]`, and `expect_metric` accepts all four. Records carry `scale`, `summary` (label) and `series_start_year` (to re-read the same cached series for a plot). Live on 2026-11-11: Amsterdam 2016-2025 yearly gave a 30-year 1991-2020 baseline (mean 777 mm, band 642-918 mm; 2023 1,101 mm and 2024 1,043 mm above it); Zhengzhou July 2021 monthly was 286 mm, rank 21 of 77 Julys, above the 80-280 mm band (the same numbers the monthly plot showed). Eval: the 4 new `scale-*` cases passed (the model chose yearly, yearly, monthly and daily wet) and the 8 existing rainfall/dry cases still passed (61.6k tokens); `dry-year-madrid` kept metric dry, `rainfall-and-literature` now used monthly (no scale is asserted there). One run each: variance not measured. Open-Meteo returned HTTP 503 for about 3 minutes during the check; the retries and the error message worked.
- Nearby candidates (added 2026-11-11): before the dominance test, `resolve_location` merges each candidate within `rainfall.same_place_km` (25 km, about one ERA5 cell) of a larger kept one. Reason: the geocoder lists a city's districts as separate places ("Amsterdam" 741,636 came with 6 districts within ~10 km, the largest New West 151,677, only 4.9x smaller, plus a heliport), so Amsterdam was ambiguous and no region or country code could fix it. Merged candidates are still listed in `other_matches` and the caveat; the ambiguity error lists only the kept ones. Two places of similar size more than 25 km apart stay ambiguous.

UI facts (M9; Gradio 6.31, plotly 7.1; smoke-tested live on 2026-11-03):
- `uv run research-agent ui [--host 127.0.0.1] [--port 7860]` serves three tabs: Ask, Search, Rainfall. Without the extra it exits 1 and says `uv sync --extra ui`. The extra (`[project.optional-dependencies] ui`) is not installed by CI (`uv sync --locked`), so `tests/test_ui_plots.py` and the app tests skip there (`importorskip`); `test_ui_handlers.py` and `test_climatology.py` run everywhere. To run all of them locally: `uv run --extra ui pytest`.
- Every tab shows a badge and every result starts with a status line saying whether the LLM was used. Search and Rainfall make no model call. Ask shows the model, calls, tokens, searches, rainfall/ENSO analyses and the tools by turn.
- `RainfallClient.analyze` returns `(record, Archive)`; the agent loop discards the archive (the model sees statistics, not daily values). `fetch_archive(location, start_year, end_year)` re-reads the same series from the cache without geocoding again; the Ask tab uses it to plot the first rainfall analysis an answer cites (other analyses are listed as not plotted).
- Plot (`ui/plots.py`), three views (Daily, Monthly and, since 2026-11-11, Yearly: the analysed period's annual totals against the baseline band and mean, `yearly_figure`) chosen by a toggle on the Rainfall tab; the Rainfall tab's view follows its Time scale radio, and the Ask tab's plot uses the view of the cited analysis's scale (`handlers.view_for`). Switching redraws the last analysis from the record and daily series kept in `gr.State` for the session (`handlers.replot`; no request, no re-analysis); the plot-year and smoothing inputs apply when Analyse is pressed or the view is switched. The Ask tab has the same toggle: `ask` takes the selected view for its first plot and returns `(record, archive)` of the first rainfall analysis the answer cites, `replot` redraws it (15-day window). Since 2026-11-11 the Ask tab also has a "Year to plot" box: after an answer it shows the year plotted (`handlers.plotted_year`, the default year), and Enter or leaving the box redraws that year from the stored series (no request, no tokens). The year is never taken from the model's text; the "only R1 is plotted, N other analyses not" remark sits in the answer text, not the plot note, because a redraw replaces the note. Plotted year: the one typed, else the event date's year, else the dry `event_year`, else the latest valid year.
- Daily view: one year of daily ERA5 values (bars) against a smoothed day-of-year baseline (line). Baseline (`rainfall/climatology.py`): for each calendar day (366 slots, 29 February from leap years only) the mean over the valid years of `baseline_start_year`-`baseline_end_year` that the daily series covers, then a centred circular moving window (default 15 days, odd, 1 to 365; a slider in the UI, not in the TOML). Missing days are skipped, never zero. It needs `min_years` valid baseline years, otherwise bars only plus a note. If the series covers only part of the baseline period, the line is drawn from the years available and the note says how many. It is a mean that a few heavy days dominate, so neither a typical day nor an extreme threshold, and a different statistic from the annual baseline of Rx1day or CDD.
- Monthly view (`climatology.monthly_totals`, `monthly_baseline`, `ui/plots.monthly_figure`): the chosen year's 12 monthly totals (bars) in front of a band from the 10th to the 90th percentile of each month's total over the valid baseline years (`BAND_LOW_PERCENTILE`, `BAND_HIGH_PERCENTILE`; linear interpolation, `statistics.quantiles(method="inclusive")`), drawn as floating bars because the months are discrete. A month total needs `min_year_coverage` of its days with data (so a total can understate by up to the missing share; a missing day is never zero) and an incomplete month, such as the current one, is left blank. A month's band needs at least `min_years` totals, and the whole band needs `min_years` valid baseline years, otherwise bars only plus a note. By construction about 20% of years fall outside a 10-90 band; the note says so. Live check (Zhengzhou 2021, baseline 1991-2020): July 286 mm against a band of 80-280, August 356 against 60-253, September 276 against 22-171, so the wet season stands out more than the single 79 mm day does in the daily view.
- Safety: titles, abstracts and model text are escaped before they reach Markdown (`handlers.md`); only `http(s)` links are clickable. The server binds to 127.0.0.1 by default; Gradio analytics are off (`analytics_enabled=False`); events queue one at a time, which also caps parallel token spend. There is no authentication: anyone who can reach the port can spend the OpenAI budget, so set `UI_USERNAME`/`UI_PASSWORD` (see Container) before exposing it.
- Ask tab conversations (M10): one `InMemorySessionStore` per app (`build_app`), a `gr.State` session ID per browser tab, `handlers.ask_turn` (returns the session ID and a "Conversation so far" transcript), a "New conversation" button. `handlers.ask` is the one-turn form. Not yet tried in a browser with the real model.
- Map point input (added 2026-11-12, Rainfall tab only): a "Location" radio (City / Point on map). The city fields are unchanged. A Leaflet map in `gr.HTML` (`head` loads Leaflet 1.9.4 from unpkg with SRI hashes; `js_on_load` sends `trigger('click', {lat, lon})`) fills Latitude/Longitude boxes (also typeable) and switches to point mode (`app._map_click`, `handlers.point_from_click`, 4 decimals). Gradio's default `gr.HTML` already has a `js_on_load`, so tests tell the map apart by its `head`. `RainfallClient.analyze_point(lat, lon, ...)` skips geocoding and shares everything after it with `analyze` (`_analyze_location`); coordinates are validated before any request (finite, within +/-90 and +/-180), and a caveat names the ERA5 grid cell used. The point's `Location` has `id=0` and the coordinates as its name. The same 4-decimal coordinates give the same archive cache key as a city. The viewer's browser loads Leaflet from unpkg and tiles from tile.openstreetmap.org (OSM tile usage policy; fine for light use). Not offered to the agent, the Ask tab or the CLI; polygons/areas are not built (they would need a cap in grid cells and an area-average rule). Live on 2026-11-12 through the app's HTTP API: the point at Zhengzhou's coordinates gave the city's numbers (79.0 mm on 2021-07-20, grid cell 34.75, 113.75) from the city's cache entry; monthly July 2021 286.2 mm; an ocean point (0, -140) yearly worked; a missing coordinate is an error before any request. Not yet tried by clicking in a browser.
- Known gaps: the Ask tab was tested with a stubbed agent, offline and through the running app (answer, plot and Daily/Monthly redraw checked; no live model run, it costs tokens); the plot was checked by its numbers (79.0 mm on 2021-07-20 for Zhengzhou, baseline 6.2 mm/day that day), not by eye; `handlers.py` imports the private `cli._run_agent` and `cli.format_trace`.

Container (added 2026-11-11; image built, served on localhost:8080 and deployed to Cloud Run):
- `Dockerfile` (python:3.12-slim, uv, `uv sync --locked --extra ui --no-dev`) runs `research-agent ui --host 0.0.0.0 --port ${PORT:-8080}` as non-root user `app`. Cloud Run sets `PORT`; on Azure Container Apps set the ingress target port to 8080. Config and `.cache/` are relative to `/app`.
- `.dockerignore` excludes `.env`, `secrets/`, `.cache/`, `.git`, tests and evals. Keys are passed as environment variables or platform secrets, never baked into the image (env vars take priority over `.env`).
- Gradio logs `http://0.0.0.0:8080`, which browsers cannot open: use `http://localhost:8080`.
- Local check: `docker build -t research-agent:local .` then `docker run --rm -p 8080:8080 --env-file .env research-agent:local`.
- Optional login: set `UI_USERNAME` and `UI_PASSWORD` in the environment or `.env` (both or neither; never a flag; read through `read_env` since 2026-11-11, before that `.env` was ignored) and `ui` passes them to Gradio's `auth`. `ui --share` also opens a temporary public `*.gradio.live` link to the local server (Gradio's tunnel) and refuses to start without the login; the local server still binds to `--host`. Not tried live (it needs Gradio's tunnel client, downloaded on first use). With neither set and a non-local host it prints a warning. Basic Gradio auth is a single shared password with no lockout beyond Gradio's own, so it is a deterrent, not strong security: use platform auth too where it exists (Cloud Run IAM/IAP, Azure built-in auth) and cap max replicas, since each replica can spend tokens.
- Cloud Run (project `test-research-514407`, `europe-west1`, service `research-agent`, private, max 1 instance, `OPENAI_API_KEY` from Secret Manager) deployed 2026-11-11 with `gcloud run deploy --source .`. Cloud Build's builder has no BuildKit, so the Dockerfile must not use `RUN --mount`. Reach it with `gcloud run services proxy research-agent --region europe-west1`. Ask tab with the real key not yet tested there. Render is being considered as an alternative (no private option, so set the login).
- The cache is lost on restart or scale-out (only costs repeated calls). The deployment platform is still undecided; the image only prepares for it.

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
- Memory across conversations. (Since M10 a conversation keeps its own turns in process memory, by explicit request; nothing is persisted or shared between sessions.)
- HTTP API. (The optional local Gradio web UI was added in M9 by explicit request; hosting it publicly is still undecided, see Open decisions.)
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
- Only `research_agent.*` loggers follow the flag. `httpx`, `openai` and friends stay at WARNING because their debug output can contain full prompts (`opentelemetry` too). The question text is never logged; it is sent to the tracing backend when tracing is on, unless `RESEARCH_AGENT_TRACE_CONTENT=false` (see Tracing).
- `REDACTOR` (`logging_setup.py`) replaces any loaded API key with `<redacted>` in log messages and tracebacks. It is fed by `_credentials_or_exit`; any new code path that loads credentials must go through that function.
- Typer is configured with `pretty_exceptions_enable=False` and `pretty_exceptions_show_locals=False`: Typer's default crash output prints local variables, which can include API keys (verified). Do not turn this back on.
- Exit codes: 0 success; 1 any failure (expected failures print one `error: ...` line; unexpected exceptions print one redacted `error: unexpected failure (...)` line, with a traceback only under `-vv`); 2 command-line usage error (Click). `eval` exits 1 if any case fails.

## Conversation facts (M10, added 2026-11-11)

- `research-agent chat` (CLI) and the UI Ask tab support follow-ups. `ask` and `eval` stay single-turn and unchanged; with no context the prompt is byte-identical (tested).
- Per turn: `ConversationService.ask(session_id, question)` loads the session, `build_context`, runs the agent with a `TurnContext`, `apply_turn`, and stores the new session. A failed turn leaves the session unchanged. An unknown or expired ID starts a new session (never another one's). Each turn keeps its own budgets, error mapping and citation validation.
- Task state (`TaskState`: places, period, event date or year, metrics, sources, searches, pending clarification) is built only from tool events (`AgentResult.events`), never from model text. The model labels the turn in its JSON (`relation`: continue, update, compare, new_topic; only asked in conversation mode). `new_topic` archives the topic as one line and starts a new state; `compare` adds to the state; `update`/`continue` replace only what the turn used. A missing label counts as `update` (`relation_source="default"`). An ambiguous place sets `pending` (with candidates) until a rainfall call resolves one.
- Context (`[conversation] max_context_tokens`, 2000) is filled in priority order: task state (always), the last turn, the current topic's evidence index (ID and label only), other recent turns (`recent_turns`), code-written one-line summaries of older turns and archived topics. Earlier answers are sent as claim digests marked "model-written, not evidence". Tool output is never resent. Left-out items are counted and reported as a limitation. Headers and separators are costed, so the context never exceeds the budget unless the task state alone does (tested over a range of budgets). The context is added to the turn's first token estimate.
- Evidence reuse: stored items (`EvidenceItem`, at most `max_stored_evidence`, oldest dropped) can be shown again with the `recall_evidence` tool (offered only when something is stored; at most `max_recall_per_call` IDs). It costs context tokens and a model turn, but no search or analysis budget and no network. Only items shown in the current turn (new or recalled) are citable; an ID that appears only in an earlier answer is rejected and its claim downgraded (tested). `R…` IDs continue across a session (`next_data_number`). Reused items add a limitation, and recalled works do not trigger the "no search was run" note.
- Sessions: `InMemorySessionStore` (thread-safe, `max_sessions` LRU, `session_ttl_minutes` idle expiry, lost on restart). A persistent store would implement `SessionStore` and rebuild `EvidenceItem.record` from `EvidenceItem.ref` (tool and arguments, mostly a cache hit).
- Tracing: one root span per turn (`chat.turn` or `ui.ask`) with `langfuse.session.id`, the turn number, relation and source, context tokens, turns sent, omitted, recallable and recalled counts.
- Live check on 2026-11-11 (CLI, 3 turns, about 22k tokens in total):
  - "Amsterdam, last 10 years": Amsterdam resolved and was analysed.
  - "What about Rotterdam?": Rotterdam was analysed with the same period and metric, and Amsterdam was recalled. The model labelled it compare, not update.
  - "Compare the two": recalls only, no new analysis.
  - Context sent: about 270 and 470 tokens.
- Known gaps: the relation label is the model's call and is not evaluated by eval cases (no multi-turn eval yet); there is no per-session token cap; the UI conversation has not been tried live in a browser.

## Tracing (added 2026-11-11)

- Off unless configured; then every `ask` run, UI request (Ask, Search, Rainfall) and eval case is one trace. Install with `uv sync --extra tracing` (the Docker image includes it). Plain OpenTelemetry API + OTLP/HTTP exporter, not the Langfuse SDK.
- Langfuse: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (all three or none; no default host, so no region is chosen for you), from the environment or `.env`. Spans go to `{host}/api/public/otel/v1/traces` with `Authorization: Basic base64(pk:sk)` and `x-langfuse-ingestion-version: 4` (without it Langfuse may delay ingestion by up to 10 min; from langfuse.com/integrations/native/opentelemetry, read 2026-11-11). Other backends: `OTEL_EXPORTER_OTLP_ENDPOINT`/`..._TRACES_ENDPOINT` + `..._HEADERS`, process environment only (the SDK reads them; `.env` is ignored for these). Both kinds set at once is an error.
- Spans: root `ask` / `ui.ask` / `ui.search` / `ui.rainfall` / `eval.case` (tags: cli/ui/eval and the case ID) > `invoke_agent` (Langfuse type agent; RunStats as `research_agent.*`) > `chat` (generation; `gen_ai.usage.*`, turn, `gen_ai.request.model` set by `OpenAIClient`) and `execute_tool {name}` (tool; `research_agent.tool.error` and ERROR status when the result starts with `Error:`).
- Content (question, answer summary, full model messages without `provider_state`, tool arguments and results) is attached by default, by the owner's choice; `RESEARCH_AGENT_TRACE_CONTENT=false` keeps metadata only. Exceptions set ERROR status with `REDACTOR`-redacted text; the SDK's own exception recording is off (it stores unredacted text and tracebacks). Langfuse and OTLP header values are added to `REDACTOR`.
- Spans are batched and flushed by the SDK at process exit. Tests use an in-memory exporter; `tests/conftest.py` removes all tracing variables, so pytest and CI never export.
- Verified: offline tests (span tree, types, tokens, content on/off, redaction, settings); one span exported to a local fake endpoint arrived on the Langfuse path with both headers. Not yet verified against a real Langfuse project: trace rendering and cost (Langfuse prices from model + tokens; `gpt-6-luna` probably needs a custom model price in the project settings).

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
- CI (`.github/workflows/ci.yml`) runs on pull requests to `main` and on every push to `main`: `uv sync --locked`, `ruff check`, `pytest`, on Ubuntu and Windows with Python 3.12 and 3.13, plus a `docker` job that builds the image (not pushed) and checks the UI answers HTTP on port 8080 within 60 s, with no keys. Runs on `main` are never cancelled by later pushes. It has no secrets and must stay offline: never add the `live` marker, `eval` or API keys to it. Actions are pinned by commit SHA with the version in a comment; update them deliberately. The `CI passed` job is the single check to require in branch protection. Keep `uv.lock` committed and current, or `--locked` fails the job.
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
uv run research-agent chat                              # follow-up questions; empty line ends
uv run research-agent -v ask "..."                      # progress log on stderr
uv run research-agent check-credentials                 # never prints key values
uv sync --extra ui                                      # optional: Gradio + plotly
uv run research-agent ui                                # web interface on http://127.0.0.1:7860
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
| M8 (done) | El Nino / La Nina events: `analyze_enso_events` (NOAA CPC RONI, strict parse, event rule, strength by peak, ongoing / not-yet-an-event / record-start flags), shared `R…` citations and rainfall budget, `enso` CLI command, `expect_tool` / `forbid_tool` eval keys, 5 eval cases | Offline tests pass (mutation spot-checks: 23/23 caught); `enso` prints events and is served from the cache on the second run; events re-derived independently match; the model picks the ENSO tool for event questions and the literature for a methods question (live eval, 26/26); fabricated `R…` IDs are still rejected |
| M7 (done) | Extreme dry analysis: `metric=dry` (CDD, longest unbroken dry spells, driest years, year and spell focus) on the same tool, renamed `analyze_rainfall_extremes`; `expect_metric` eval key; 4 eval cases | Offline tests pass; `rainfall --metric dry` prints statistics and is served from the cache after a wet run; the model picks `dry` for dry questions and `wet` for heavy-rain ones (live eval); fabricated `R…` IDs are still rejected |

| M9 (done) | Optional Gradio web interface (Ask, Search, Rainfall tabs, LLM-used badges and status lines), `ui` CLI command, `analyze` returns the daily `Archive`, smoothed day-of-year baseline and daily plot | Offline tests pass with and without the extra; the app serves and Search and Rainfall (wet, dry, ambiguous city) answer through its HTTP API; the plot's traces carry the expected daily values and baseline |

Priorities, in order: correct citations, budget enforcement, tests, clarity, simplicity.
