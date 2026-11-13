# GeoAI Research Agent

A command-line research agent that answers questions about GeoAI and remote-sensing literature. It searches [OpenAlex](https://openalex.org), reads the titles and abstracts it finds, and writes a short answer in which every statement is either **evidence** (cited to a work it actually retrieved) or **interpretation** (its own synthesis), followed by references and an explicit list of limitations.

It can also quantify **historical extreme rainfall and extreme dry spells in a city** (Rx1day, consecutive dry days and driest years, from ERA5 reanalysis), compare an event or a year with the record, and explain it with the literature. It can also **find El Niño and La Niña events** from NOAA's official ENSO index (RONI). The model decides, per question, whether the literature, a computed analysis or both is the right source.

It is deliberately small and cheap to run: one LLM agent with two tools, hard caps on searches, turns and tokens, and no full-text PDFs, vector databases or agent frameworks.

## What you get

Example output (abridged):

```
$ uv run research-agent ask "Which deep learning methods are used for crop type mapping from Sentinel-2 time series?"

Answer
- (evidence) Methods reported for crop-type mapping from Sentinel-2 time series include 1D CNNs,
  LSTMs and GRUs, as well as hybrid LSTM-CNN and GRU-CNN models. [W3183470088]
- (evidence) A separate study used an LSTM to identify crop types early in the growing season. [W4319319540]
- (evidence) Other approaches include multilayer perceptrons and CNN variants ... [W3122949800]

References
[W3183470088] Hongwei Zhao, Sibo Duan, Jia Liu et al. (2021). Evaluation of Five Deep Learning Models for
    Crop Type Mapping Using Sentinel-2 Time Series Images with Missing Information.
    Remote Sensing. https://doi.org/10.3390/rs13142790
...

Limitations
- This summary is based on abstracts from a narrow literature search, so it may not cover all methods.
- Evidence is limited to titles, abstracts and metadata from OpenAlex; full texts and other databases were not searched.
- Only the top-ranked results of each search were read (for example 8 of 63 matches); relevant work may have been missed.
- 2 of 8 retrieved work(s) have no abstract and could be judged by title only.

run: 3 model turn(s), 2 search(es) (0 from cache), 5563 tokens (5299 in / 264 out)
```

The answer goes to stdout; the `run:` line goes to stderr.

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
- An **OpenAI API key** (required for `ask` and `eval`)
- An **OpenAlex API key** (optional, but recommended; see [Credentials](#credentials))

## Setup

```bash
uv sync
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```

Edit `.env` and put in your keys (one per line, no quotes):

```
OPENALEX_API_KEY=...
OPENAI_API_KEY=...
```

Check the setup. This is an offline check that prints only `set` or `not set`, never the key values:

```bash
uv run research-agent check-credentials
```

Run all commands from the repository root: the default config path is `configs/base.toml`, relative to the current directory.

## Usage

| Command | What it does |
|---|---|
| `uv run research-agent ask "<question>"` | Answer a question from the literature and/or the rainfall analysis (uses the LLM, OpenAlex and Open-Meteo) |
| `uv run research-agent search "<query>" [--year-from 2022] [-n 5] [--abstracts]` | Search OpenAlex only, with no LLM, and print the results |
| `uv run research-agent rainfall "<city>" [--country-code CN] [--region Henan] [--event-date 2021-07-20] [--start-year 1950] [--end-year 2025] [--metric dry] [--event-year 2022]` | Rainfall-extremes analysis only, with no LLM. Default (wet): Rx1day per year, baseline, largest events, how extreme one day was. `--metric dry`: dry spells, driest years, how dry a year or dry spell was |
| `uv run research-agent enso [--phase el_nino\|la_nina\|both] [--start-year 1990] [--end-year 2020] [--year 2010]` | El Niño / La Niña events from NOAA CPC's RONI only, with no LLM: first and last season, length, peak, strength; `--year` also shows that year's seasonal values and overlapping events |
| `uv run research-agent ui [--host 127.0.0.1] [--port 7860]` | Web interface with Ask, Search and Rainfall tabs; each says whether the LLM is used, and Rainfall plots one year either as daily values against a smoothed day-of-year mean or as monthly totals against the baseline 10th-90th percentile band (a toggle). Needs `uv sync --extra ui`; local use only, it has no login |
| `uv run research-agent eval run --live [--only <case-id>]` | Run the evaluation cases in `evals/cases.toml` within a run budget (costs tokens; without `--live` it only prints the plan) |
| `uv run research-agent check-credentials [--require-openai]` | Check that keys are present and well formed |
| `uv run research-agent show-config` | Validate `configs/base.toml` and print the resource limits |

Add `-v` for progress logging on stderr (each model turn, token counts, each search) and `-vv` for debug detail. The flag goes before the command:

```bash
uv run research-agent -v ask "How are vision transformers used for land cover classification?"
```

**Exit codes:** 0 success, 1 any failure (one `error: ...` line explains it), 2 command-line usage error. `eval` exits 1 if any case fails.

## How it works

```
CLI -> agent loop -> LLM (OpenAI)   <-->  search_literature tool        -> OpenAlex (+ disk cache)
                |                           <-->  analyze_rainfall_extremes tool -> Open-Meteo geocoding + ERA5 archive (+ disk cache)
                |                           <-->  analyze_enso_events tool       -> NOAA CPC RONI text file (+ disk cache)
                |
                +-> structured answer: claims (evidence / interpretation), references, limitations
```

1. The model decides which tool to use, if any, and with what arguments: `search_literature` for methods and explanations, `analyze_rainfall_extremes` for numbers about past heavy rain (`metric=wet`) or dry spells (`metric=dry`) in a city, `analyze_enso_events` for which years or events were El Niño or La Niña, or a combination.
2. Results are formatted deterministically and sent back to the model. Retrieval and all rainfall arithmetic are plain Python, with no AI. Literature results carry work IDs (`W123`); rainfall results carry analysis IDs (`R1`).
3. The model replies with JSON: claims, each tagged as evidence or interpretation with the IDs (works or rainfall analyses) it relies on.
4. The code, not the model, then enforces the contract:
   - **Citations are checked.** A citation to a work or rainfall analysis that was never retrieved or computed is removed and reported. An "evidence" claim left with no valid citation is downgraded to interpretation.
   - **References are built from retrieved works and computed analyses**, never from text the model wrote.
   - **Limitations are always added**: abstracts-only evidence, OpenAlex-only coverage, thin or dated evidence, missing or truncated abstracts, how much of the result set was actually read, and any budget that was hit. Rainfall analyses add their own: ERA5 is model output, not gauge data; the grid cell is about 25 km; Rx1day ignores multi-day accumulation; excluded years; short records; other places with the same name.

### Extreme rainfall and dry spells

`analyze_rainfall_extremes` geocodes the city with [Open-Meteo](https://open-meteo.com), downloads daily precipitation from the ERA5 reanalysis (model pinned to `era5`, local calendar days, default 1950 to the last complete year) and computes, in Python: the largest one-day total of each year (Rx1day), a 1991-2020 baseline mean and median, the largest events on record, and, for a given `event_date`, its rank among all days and among the yearly maxima. Each result states its provenance (dataset, request URL, grid cell actually used, licence).

With `metric=dry` the same data gives, again in Python: the longest run of consecutive dry days (CDD; a dry day has less than 1 mm) in each year and a 1991-2020 baseline for it; the longest unbroken dry spells on record with dates (not split at New Year); the driest years by annual total precipitation (all days); and, for an `event_year`, that year's total, rank (1 = driest) and CDD, or, for an `event_date`, the dry spell containing that day and its rank among all spells. The geocoding and the data come from the same cached request as the wet analysis, so a dry analysis of a city already analysed makes no network call (it still counts as one of the 3 rainfall analyses).

- **A dry spell is not a drought.** It is days under 1 mm of rain: no evaporation, soil moisture or accumulated deficit. SPI/SPEI are not computed, and the model is told not to call a dry spell or dry year a drought unless a retrieved paper says so.
- **ERA5 may shorten dry spells.** Regional studies (Spain, Vietnam) find that ERA5 produces too many light-rain days, which breaks dry runs; absolute lengths here may be underestimates and are reported with that caveat. Yearly CDD restarts on 1 January (the unbroken list does not), and where there is a distinct dry season CDD mostly reflects its length.
- **The period defaults to 1950 to the last complete year.** The model is told to leave it unset unless the question names one, so rankings stay comparable between questions.

- **Ambiguous names are never guessed.** If several places match and none has at least 10 times the population of the runner-up, the tool returns the candidates and no analysis is run; the answer says the location is ambiguous. A clearly dominant match (for example Paris, France) is used and the alternatives are listed in the limitations.
- **Short or gappy records withhold ranks.** Years with under 90% of days, and the current year, are excluded and reported; fewer than 20 valid years means no ranks or baseline comparison.
- **Open-Meteo terms:** the free tier is for non-commercial use, with attribution (CC BY 4.0) and a daily call cap. Commercial use needs a paid Open-Meteo plan.
- **ERA5 smooths extremes.** A grid-cell daily total usually understates what a single station records. For example, ERA5 puts 20 July 2021 in Zhengzhou, an event widely described as exceptional, at roughly 80 mm and only about 22nd among 76 yearly maxima; a different grid cell or reanalysis product gives quite different values (an unpinned Open-Meteo request returned about 150 mm for the same day, which is why the model is pinned). Read rankings as "within this reanalysis at this grid cell", not as the severity people experienced.

### El Niño and La Niña events

`analyze_enso_events` downloads NOAA CPC's [RONI table](https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt) (Relative Oceanic Niño Index, 1950 to now, one small text file, no key) and finds events in Python with NOAA's rule: the index at or beyond ±0.5 (a value of exactly 0.5 counts) for at least 5 consecutive overlapping 3-month seasons. For each event it gives the first and last season, the number of seasons, the peak value and season, and a strength label; with `year` it also gives that year's seasonal values and any overlapping event. RONI is NOAA's official ENSO index since February 2026; the legacy ONI is still published and is what many older papers use, so a paper's list of events can differ from this one.

- **Literature or analysis is the model's choice.** The prompt sends "which years/events, how strong or long" questions to this tool and causes, impacts and "what did the paper count" to the literature search; a question that needs both uses both. Whether a real model chooses well is judged by the live eval cases (`expect_tool` / `forbid_tool` on the tool trace), not by the offline tests.
- **The latest values are not final.** The last seasons can be revised, and an event reaching the end of the record is marked ongoing. Conditions at the threshold for fewer than 5 seasons (as in mid-2026) are reported as "not yet an event", never as an event. An event that starts at the first season of the file (La Niña, DJF 1950) may have begun earlier and is flagged.
- **Strength labels are a convention, not NOAA's.** By peak value: weak from 0.5, moderate from 1.0, strong from 1.5, very strong from 2.0. A stricter, also common version needs the class to be reached for 3 consecutive seasons; it would lower the label of 12 of the 44 events in the 2026-11-02 record (for example the 1991-92 El Niño from very strong to strong). The result says so.
- **Strict parsing.** A missing, duplicated or out-of-order season, an implausible value (for example a -99.9 missing-value marker) or an unexpected header makes the download fail with a clear error instead of producing wrong events, and a file that does not parse is never cached. The file does not say which sea-surface-temperature dataset version it uses, and the output does not claim one.
- **No licence statement was found** on the NOAA pages I read; the source URL is always reported.
- **Not covered:** rainfall or dry spells by ENSO phase, other indices (ONI, MEI, SOI), forecasts, local impacts.

### Resource limits

All limits are in `configs/base.toml` and enforced in code, not just in the prompt:

| Limit | Default |
|---|---|
| Model turns per question | 5 |
| Searches per question | 3 |
| Results per search | 8 |
| Abstract length | 1,200 characters |
| Retrieved text sent to the model | about 6,000 tokens |
| Output tokens per model call | 800 |
| Total tokens per question | about 20,000 |
| Question length | 2,000 characters |
| Rainfall and ENSO analyses per question (one shared limit) | 3 |

Rainfall and ENSO results count against the same retrieved-text limit as literature. When every tool budget is spent, the model's next turn has tools disabled so it must answer with the evidence it already has, and the answer's limitations say which limit was hit. A typical question uses about 5,000 to 11,000 tokens.

### Credentials

- Keys come from the process environment first, then from `.env`. `.env` is git-ignored; `.env.example` is the tracked template with placeholders only.
- To keep `.env` somewhere else (for example outside a synced folder), pass `--env-file <path>` or set `RESEARCH_AGENT_ENV_FILE`.
- Keys are wrapped so they cannot appear in `repr`, logs or error messages, and any loaded key is also redacted from log output and tracebacks.
- Without an OpenAlex key, searches share a small anonymous daily budget with everyone on your network's IP address. A free key gives your own larger budget. Get one at <https://openalex.org/settings/api>.

## Evaluation

`evals/cases.toml` holds 26 cases (the 5 ENSO cases are listed after this paragraph): 10 on-topic literature questions, one off-topic question, one prompt-injection attempt, 5 extreme-rainfall cases (event ranking, largest events, rainfall plus literature, an ambiguous city, years before the archive starts) and 4 dry-spell cases (longest spells, how dry a year was, dry spell plus literature, wet and dry together). Cases can state `expect_metric`, which checks the model's choice of `wet` or `dry` on the tool trace, and `expect_tool` / `forbid_tool`, which check which tool it requested. The 5 ENSO cases: events since 1990, was 2010 El Niño or La Niña, a literature-only question that must not call the ENSO tool, strongest La Niña events plus their impacts in the literature, and current conditions (time-dependent: it describes the record as of 2026-11-02). Answers vary between runs, so each case states expected **properties** (it searched, cited enough works, mentions a topic keyword, did not leak the system prompt) and not exact text. Every case is also checked against invariants: structured answer, limitations present, every citation resolves to a reference, budgets respected.

```bash
uv run research-agent eval list                          # free: cases with their split and category
uv run research-agent eval run                           # free: prints the plan (cases, worst-case tokens, budget)
uv run research-agent eval run --live                    # the dev split with the real model (about 6.5k tokens per case)
uv run research-agent eval run --live --only crop-type-dl --max-tokens 30000
uv run research-agent eval score evals/runs/<run-id>     # free: score stored records again, offline
uv run research-agent eval replay evals/runs/<run-id>    # free: the recorded model replies through the current code
uv run research-agent eval run --live --mode fixed       # baseline: one search with the question, then one answer
uv run research-agent eval compare <run-a> <run-b>       # free: case-by-case differences between two runs
uv run research-agent eval run --live --suite conversations  # multi-turn conversations
uv run --extra judge research-agent eval judge evals/runs/<run-id> --live  # LLM judge (cached)
uv run research-agent eval import-langfuse [--all]       # Langfuse traces as run records
```

Each case is marked `split = "test"` (held out: run only to measure, with `--split test`) or dev, and has a `category`. A paid run stops before any model call that would exceed the run budget (`[eval]` in `configs/base.toml`: model calls, tokens, and a cost cap once prices are filled in) and writes one JSON record per case to `evals/runs/<run-id>/`, so the results can be scored again without running the agent. Tool responses are kept in `evals/tool_cache/` (no expiry; `--frozen-tools` also disables the network).

`eval score` (and `eval run --live` when it finishes) also writes `score.json` and `report.md` into the run folder. They hold metrics that are reported but never fail a case:
- tool errors by kind, and redundant or uncited tool calls;
- whether the model stopped by itself or a budget forced the answer;
- numbers in evidence claims that the cited text does not contain;
- claims whose words barely appear in the cited text;
- key-point coverage;
- retrieval precision and recall against `evals/qrels.toml`, with the works not yet labelled listed so you can label them;
- tokens, time and cost.

Cases can also state `expect_numbers` (verified values that must appear in a tool result and in a claim), `max_tool_calls`, and `key_points` with `min_key_points`; these do fail a case.

`eval replay` re-runs a stored run through the current code without calling a model or needing a key:
- Each request must match the recorded one, and the tools answer from `evals/tool_cache/` with the network off. If the prompt, a tool definition or a tool's output changed, the case stops as **diverged**, with the first difference.
- Otherwise the new answer is compared with the recorded one and marked **same** or **changed** (changed is reported only).
- Use it to check that a code change after the model's decisions (parsing, citations, limitations) does what you expect, for free.
- It needs the tool cache of the machine that made the run.

`eval run --mode fixed` is a baseline for the literature cases:
- one search with the question as asked (or a case's `fixed_query`), then one model call with tools off;
- the same prompt and the same answer checks as the agent, so only the choice of searches differs.

`eval run --suite conversations` asks the conversations in `evals/conversations.toml` turn by turn in one session each, as `chat` does. Each turn is checked like a case and also on the model's relation label (continue, update, compare, new_topic), whether it recalled earlier evidence instead of repeating an analysis, and the places, metrics and pending clarification in the task state the code kept. Score, replay and compare work on these runs too.

`eval judge` (optional, needs `uv sync --extra judge`, `CLAUDE_API_KEY` and `CLAUDE_MODEL` or the `[eval.judge]` model) asks an Anthropic model (Claude Haiku 5.5 by default) to rate each evidence claim against the text it cites: supported, partial or unsupported, with a reason. Verdicts are cached in `evals/judgements/`, so an unchanged claim is never judged twice, and they are reported by `eval score` (never a case failure): `report.md` gets a Judge section with tables by run, category, split and case, and a score (supported 1, partial 0.5, unsupported 0, averaged over judged claims). `--sample 10` writes a calibration sheet for checking the judge yourself.

`eval import-langfuse` reads traces back from Langfuse (the `LANGFUSE_*` keys, the v2 observations API) and turns each agent trace into a run record: eval traces by default, real `ask`, UI and chat traces with `--all`. Score, the LLM judge and compare then work on them; replay does not (a trace holds no request hashes).

`eval compare A B` pairs the cases of two runs and counts, for each measure, how often B did better, the same or worse than A. Measures: pass rate, unsupported numbers, low-support claims, key points, precision and recall, tokens, model calls, time and cost.

The cases are a draft; edit them to match the questions you care about.

## Development

```bash
uv run pytest                  # offline: no network, no API cost
uv run pytest -m live          # opt-in: real OpenAlex, Open-Meteo and OpenAI calls (a few cents at most)
uv run ruff check .
```

Every pull request to `main` runs lint and the offline tests on GitHub Actions (`.github/workflows/ci.yml`: Ubuntu and Windows, Python 3.12 and 3.13). CI uses no secrets and never calls OpenAI or OpenAlex, so live tests and `eval` are not run there; run those yourself before merging changes that touch the agent, the prompt or the adapters.

The default test run uses a scripted stub LLM and recorded OpenAlex responses. See [`CLAUDE.md`](CLAUDE.md) for the architecture, engineering principles, milestone history and detailed behavioural notes.

## Known limitations

- **Abstracts only.** It never reads full texts, and it searches only OpenAlex. Some works have no abstract.
- **Rainfall is reanalysis, one city at a time.** No gauge data, no return periods, drought indices or trends, no latitude/longitude input, and data only from 1940. The numbers in a claim are not machine-checked against the cited analysis: the code guarantees the citation exists, not that the sentence quotes it correctly.
- **Not restricted to GeoAI.** For an off-topic question it still searches OpenAlex (which covers every field) and answers from whatever it finds. There is no scope guard yet.
- **Long keyword queries can match nothing.** OpenAlex's title-and-abstract search appears to require all terms, so the model occasionally wastes a search on an over-specific query.
- **A fluent, cited answer is not proof that a claim matches its abstract.** The code guarantees that cited works were retrieved, not that the model summarised them correctly. Read the references.
- **Your question is sent to OpenAI** and, with the default Responses API settings, may be stored by OpenAI on their side. Searches are sent to OpenAlex.
- **Model names and prices change.** The default model (`gpt-6-luna`) is set in `configs/base.toml`; check current pricing yourself.
