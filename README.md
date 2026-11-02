# GeoAI Research Agent

A command-line research agent that answers questions about GeoAI and remote-sensing literature. It searches [OpenAlex](https://openalex.org), reads the titles and abstracts it finds, and writes a short answer in which every statement is either **evidence** (cited to a work it actually retrieved) or **interpretation** (its own synthesis), followed by references and an explicit list of limitations.

It can also quantify **historical extreme rainfall in a city** (annual maximum daily precipitation, Rx1day, from ERA5 reanalysis), compare an event with the record, and explain it with the literature.

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
| `uv run research-agent rainfall "<city>" [--country-code CN] [--region Henan] [--event-date 2021-07-20] [--start-year 1950] [--end-year 2025]` | Extreme-rainfall analysis only, with no LLM: Rx1day per year, baseline, largest events, and how extreme one day was |
| `uv run research-agent eval [--only <case-id>]` | Run the evaluation cases in `evals/cases.toml` (costs tokens and searches) |
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
                |                           <-->  analyze_extreme_rainfall tool -> Open-Meteo geocoding + ERA5 archive (+ disk cache)
                |
                +-> structured answer: claims (evidence / interpretation), references, limitations
```

1. The model decides which tool to use, if any, and with what arguments: `search_literature` for methods and explanations, `analyze_extreme_rainfall` for numbers about past rainfall in a city, or both.
2. Results are formatted deterministically and sent back to the model. Retrieval and all rainfall arithmetic are plain Python, with no AI. Literature results carry work IDs (`W123`); rainfall results carry analysis IDs (`R1`).
3. The model replies with JSON: claims, each tagged as evidence or interpretation with the IDs (works or rainfall analyses) it relies on.
4. The code, not the model, then enforces the contract:
   - **Citations are checked.** A citation to a work or rainfall analysis that was never retrieved or computed is removed and reported. An "evidence" claim left with no valid citation is downgraded to interpretation.
   - **References are built from retrieved works and computed analyses**, never from text the model wrote.
   - **Limitations are always added**: abstracts-only evidence, OpenAlex-only coverage, thin or dated evidence, missing or truncated abstracts, how much of the result set was actually read, and any budget that was hit. Rainfall analyses add their own: ERA5 is model output, not gauge data; the grid cell is about 25 km; Rx1day ignores multi-day accumulation; excluded years; short records; other places with the same name.

### Extreme rainfall (Rx1day)

`analyze_extreme_rainfall` geocodes the city with [Open-Meteo](https://open-meteo.com), downloads daily precipitation from the ERA5 reanalysis (model pinned to `era5`, local calendar days, default 1950 to the last complete year) and computes, in Python: the largest one-day total of each year (Rx1day), a 1991-2020 baseline mean and median, the largest events on record, and, for a given `event_date`, its rank among all days and among the yearly maxima. Each result states its provenance (dataset, request URL, grid cell actually used, licence).

- **Ambiguous names are never guessed.** If several places match and none has at least 10 times the population of the runner-up, the tool returns the candidates and no analysis is run; the answer says the location is ambiguous. A clearly dominant match (for example Paris, France) is used and the alternatives are listed in the limitations.
- **Short or gappy records withhold ranks.** Years with under 90% of days, and the current year, are excluded and reported; fewer than 20 valid years means no ranks or baseline comparison.
- **Open-Meteo terms:** the free tier is for non-commercial use, with attribution (CC BY 4.0) and a daily call cap. Commercial use needs a paid Open-Meteo plan.
- **ERA5 smooths extremes.** A grid-cell daily total usually understates what a single station records. For example, ERA5 puts 20 July 2021 in Zhengzhou, an event widely described as exceptional, at roughly 80 mm and only about 22nd among 76 yearly maxima; a different grid cell or reanalysis product gives quite different values (an unpinned Open-Meteo request returned about 150 mm for the same day, which is why the model is pinned). Read rankings as "within this reanalysis at this grid cell", not as the severity people experienced.

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
| Rainfall analyses per question | 3 |

Rainfall results count against the same retrieved-text limit as literature. When every tool budget is spent, the model's next turn has tools disabled so it must answer with the evidence it already has, and the answer's limitations say which limit was hit. A typical question uses about 5,000 to 11,000 tokens.

### Credentials

- Keys come from the process environment first, then from `.env`. `.env` is git-ignored; `.env.example` is the tracked template with placeholders only.
- To keep `.env` somewhere else (for example outside a synced folder), pass `--env-file <path>` or set `RESEARCH_AGENT_ENV_FILE`.
- Keys are wrapped so they cannot appear in `repr`, logs or error messages, and any loaded key is also redacted from log output and tracebacks.
- Without an OpenAlex key, searches share a small anonymous daily budget with everyone on your network's IP address. A free key gives your own larger budget. Get one at <https://openalex.org/settings/api>.

## Evaluation

`evals/cases.toml` holds 17 cases: 10 on-topic literature questions, one off-topic question, one prompt-injection attempt and 5 extreme-rainfall cases (event ranking, largest events, rainfall plus literature, an ambiguous city, years before the archive starts). Answers vary between runs, so each case states expected **properties** (it searched, cited enough works, mentions a topic keyword, did not leak the system prompt) and not exact text. Every case is also checked against invariants: structured answer, limitations present, every citation resolves to a reference, budgets respected.

```bash
uv run research-agent eval                  # all cases (the 12 literature cases take about 78k tokens and 25 searches; the 5 rainfall cases about 20k tokens)
uv run research-agent eval --only crop-type-dl
```

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
- **Rainfall is reanalysis, one city at a time.** No gauge data, no return periods or trends, no latitude/longitude input, and data only from 1940. The numbers in a claim are not machine-checked against the cited analysis: the code guarantees the citation exists, not that the sentence quotes it correctly.
- **Not restricted to GeoAI.** For an off-topic question it still searches OpenAlex (which covers every field) and answers from whatever it finds. There is no scope guard yet.
- **Long keyword queries can match nothing.** OpenAlex's title-and-abstract search appears to require all terms, so the model occasionally wastes a search on an over-specific query.
- **A fluent, cited answer is not proof that a claim matches its abstract.** The code guarantees that cited works were retrieved, not that the model summarised them correctly. Read the references.
- **Your question is sent to OpenAI** and, with the default Responses API settings, may be stored by OpenAI on their side. Searches are sent to OpenAlex.
- **Model names and prices change.** The default model (`gpt-6-luna`) is set in `configs/base.toml`; check current pricing yourself.
