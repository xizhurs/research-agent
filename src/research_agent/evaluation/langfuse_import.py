"""Importing Langfuse traces as run records (``eval import-langfuse``; network, no model call).

Reads only ``GET /api/public/v2/observations`` (Langfuse's v1 trace and observation reads are
served on Langfuse Cloud only until 2026-11-16). Facts below were checked against a real project
on 2026-11-13, not only the docs:

- Basic auth with the public and secret key; ``fromStartTime``/``toStartTime`` bound every
  request; ``filter`` takes a JSON array (tags: ``arrayOptions`` / ``any of``); paging by
  ``cursor`` (``meta`` is empty when there is no next page); ``expandMetadata=attributes``.
- Metadata keys are flat dotted strings: ``attributes.research_agent.iterations``.
- ``usageDetails`` is ``{"input", "output", "total"}``; ``latency`` is in seconds.
- Our spans: root ``eval.case`` / ``eval.turn`` / ``ask`` / ``ui.ask`` / ``chat.turn`` (input =
  question, output = answer JSON), ``invoke_agent`` (run stats), ``chat`` generations (input =
  the instructions and messages, output = the reply with tool calls), one TOOL row per call
  (input = arguments, output = the result text, ``gen_ai.tool.call.id``).

``trace_record`` is pure: it turns one trace's rows into a record the scorers, the judge and
compare can use. Request hashes cannot be rebuilt from a trace, so imported records are not
replayable.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from research_agent.agent.loop import RunStats, _trace_name  # the agent's own trace naming
from research_agent.evaluation.cases import EvalCase
from research_agent.evaluation.recording import fingerprint
from research_agent.llm import ToolCall

AGENT_ROOTS = ("eval.case", "eval.turn", "ask", "ui.ask", "chat.turn")
TURN_ROOTS = ("eval.turn", "chat.turn", "ui.ask")
FIELDS = "core,basic,time,io,metadata,model,usage,metrics,trace_context"
PAGE_LIMIT = 1000  # the v2 maximum
MAX_PAGES = 50
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30.0
ROWS_GRACE = timedelta(hours=1)  # children may start after the newest root
RECALL = "recall_evidence"
SEARCH = "search_literature"

_ITEM = re.compile(r"^\[((?:R|W)\d+)\]", re.MULTILINE)
_AMBIGUOUS = re.compile(r"^Error: '.*' matches \d+ places")


class LangfuseImportError(Exception):
    """The API refused or failed (message never contains the keys)."""


def _attr(row: dict[str, Any], name: str, default: Any = None) -> Any:
    return (row.get("metadata") or {}).get(f"attributes.{name}", default)


def _json(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# --- reading -------------------------------------------------------------------------------------


class LangfuseReader:
    """Pages through ``/api/public/v2/observations`` with bounded retries on 429 and 5xx."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = http_client or httpx.Client(timeout=TIMEOUT_SECONDS)
        self._base = host.rstrip("/") + "/api/public"
        self._auth = (public_key, secret_key)
        self._sleep = sleep
        self.requests = 0

    def close(self) -> None:
        self._client.close()

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(MAX_RETRIES + 1):
            self.requests += 1
            try:
                response = self._client.get(
                    f"{self._base}/v2/observations", params=params, auth=self._auth
                )
            except httpx.HTTPError as exc:
                if attempt == MAX_RETRIES:
                    raise LangfuseImportError(f"Langfuse request failed: {exc}") from exc
                self._sleep(2.0**attempt)
                continue
            if response.status_code in (401, 403):
                raise LangfuseImportError(
                    f"Langfuse rejected the keys (HTTP {response.status_code})"
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES:
                    raise LangfuseImportError(f"Langfuse HTTP {response.status_code} after retries")
                wait = response.headers.get("Retry-After")
                self._sleep(float(wait) if wait and wait.isdigit() else 2.0**attempt)
                continue
            if response.status_code != 200:
                raise LangfuseImportError(
                    f"Langfuse HTTP {response.status_code}: {response.text[:200]}"
                )
            return response.json()
        raise LangfuseImportError("unreachable")  # pragma: no cover

    def observations(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Every row matching ``params``, following ``meta.cursor``; at most ``MAX_PAGES``."""
        cursor = None
        for _ in range(MAX_PAGES):
            page = self._get({**params, "limit": PAGE_LIMIT, **({"cursor": cursor} if cursor
                                                               else {})})  # fmt: skip
            yield from page.get("data", [])
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor:
                return
        raise LangfuseImportError(f"more than {MAX_PAGES} pages; narrow --since/--until")


def tag_filter(tag: str) -> str:
    return json.dumps([{"type": "arrayOptions", "column": "tags", "operator": "any of",
                        "value": [tag]}])  # fmt: skip


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def fetch_traces(
    reader: LangfuseReader,
    since: str,
    until: str,
    tag: str | None,
    limit: int,
    grace: timedelta = ROWS_GRACE,
) -> dict[str, list[dict[str, Any]]]:
    """The rows of the newest ``limit`` agent traces started in [since, until), by trace ID.

    Roots are listed first (newest first; the tag filter applies here, because Langfuse keeps a
    trace's tags on its root row only, checked 2026-11-13). Then the rows from the earliest to
    the latest selected root (+ ``grace`` for children that start later) are read once,
    unfiltered, and kept for the selected traces: a few requests instead of one per trace."""
    window = {"fromStartTime": since, "toStartTime": until}
    tagged = {"filter": tag_filter(tag)} if tag else {}
    wanted: dict[str, datetime] = {}
    for root in reader.observations({**window, **tagged, "isRootObservation": "true",
                                     "fields": "core,basic"}):  # fmt: skip
        if root.get("name") in AGENT_ROOTS and root["traceId"] not in wanted:
            wanted[root["traceId"]] = datetime.fromisoformat(root["startTime"])
        if len(wanted) >= limit:
            break
    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in wanted}
    if not wanted:
        return rows
    params = {"fromStartTime": _iso(min(wanted.values())),
              "toStartTime": _iso(max(wanted.values()) + grace),
              "fields": FIELDS, "expandMetadata": "attributes"}  # fmt: skip
    for row in reader.observations(params):
        if row["traceId"] in rows:
            rows[row["traceId"]].append(row)
    return rows


# --- converting ----------------------------------------------------------------------------------


def _blocks(text: str) -> dict[str, str]:
    """``[R1]``/``[W…]`` items of a tool result, each up to the next item (recall joins items
    with a blank line; an analysis result is one item)."""
    starts = [(m.start(), m.group(1)) for m in _ITEM.finditer(text)]
    out = {}
    for n, (pos, item_id) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(text)
        out.setdefault(item_id, text[pos:end].rstrip())
    return out


def _work(block: str) -> dict[str, Any]:
    """A ``[W…] Title (meta)\\nAbstract: …`` block as a record's work entry."""
    head, _, rest = block.partition("\n")
    title = head.split("] ", 1)[1] if "] " in head else head
    if title.endswith(")") and " (" in title:
        title = title.rsplit(" (", 1)[0]
    abstract = rest.removeprefix("Abstract: ").strip()
    return {"id": block[1:block.index("]")], "title": title,
            "abstract": None if abstract == "(not available)" else abstract, "year": None}


def _status(row: dict[str, Any]) -> str:
    text = row.get("output") or ""
    if _AMBIGUOUS.match(text):
        return "ambiguous"
    return "error" if _attr(row, "research_agent.tool.error") or text.startswith("Error:") else "ok"


def trace_record(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    case_for: Callable[[dict[str, Any], str], EvalCase | None],
) -> tuple[dict[str, Any] | None, str | None]:
    """One trace as a run record, or (None, why it was skipped). ``case_for(root, question)``
    finds the case or conversation turn the trace answered (None: an unmatched trace)."""
    ordered = sorted(rows, key=lambda r: (r.get("startTime") or "", r.get("type") != "SPAN"))
    root = next((r for r in ordered if not r.get("parentObservationId")), None)
    if root is None:
        return None, "no root observation (still being ingested?)"
    if root.get("name") not in AGENT_ROOTS:
        return None, f"not an agent trace (root '{root.get('name')}')"
    agent = next((r for r in ordered if r.get("name") == "invoke_agent"), None)
    question = root.get("input") or (agent or {}).get("input")
    if not question:
        return None, "no content (traced with RESEARCH_AGENT_TRACE_CONTENT=false?)"
    trace_id = root["traceId"]
    case = case_for(root, question)
    chats = [r for r in ordered if r.get("type") == "GENERATION" and r.get("name") == "chat"]
    tools = [r for r in ordered if r.get("type") == "TOOL"]
    first_input = _json(chats[0].get("input")) if chats else None
    system = (first_input[0].get("content") if isinstance(first_input, list) and first_input
              and first_input[0].get("role") == "system" else None)  # fmt: skip

    llm_calls, trace, call_turn = [], [], {}
    for turn, chat in enumerate(chats, start=1):
        reply = _json(chat.get("output")) or {}
        calls = reply.get("tool_calls") or []
        for call in calls:
            call_turn[call.get("id")] = turn
        usage = chat.get("usageDetails") or {}
        reasons = _attr(chat, "gen_ai.response.finish_reasons") or [None]
        llm_calls.append({
            "request_hash": None, "tools_hash": None, "max_output_tokens": None,
            "allow_tools": bool(_attr(chat, "research_agent.allow_tools", True)),
            "latency_s": chat.get("latency"), "stop_reason": reasons[0],
            "usage": {"input": int(usage.get("input") or 0),
                      "output": int(usage.get("output") or 0)},
            "text": reply.get("content"),
            "tool_calls": [{"id": c.get("id"), "name": c.get("name"),
                            "arguments": c.get("arguments") or {}, "error": None} for c in calls],
        })  # fmt: skip
        trace.append([_trace_name(ToolCall(c.get("id") or "", c.get("name") or "",
                                           c.get("arguments") or {})) for c in calls])  # fmt: skip

    tool_calls, data_texts, work_blocks, recalled = [], {}, {}, []
    for row in tools:
        name = _attr(row, "gen_ai.tool.name") or row.get("name")
        call_id = _attr(row, "gen_ai.tool.call.id")
        output, status = row.get("output") or "", _status(row)
        blocks = _blocks(output) if status == "ok" else {}
        own = [i for i in blocks if i.startswith("R")]
        result_id = own[0] if name not in (SEARCH, RECALL) and own else None
        arguments = _json(row.get("input")) or {}
        if name == RECALL and status == "ok":
            recalled += [i for i in arguments.get("ids", []) if i in blocks]
        for item_id, block in blocks.items():
            if item_id.startswith("R") and (name != RECALL or item_id not in data_texts):
                data_texts.setdefault(item_id, block if name != RECALL else block)
            elif item_id.startswith("W") and "already shown" not in block:
                work_blocks.setdefault(item_id, block)
        tool_calls.append({"turn": call_turn.get(call_id), "id": call_id, "name": name,
                           "arguments": arguments, "status": status, "result_id": result_id,
                           "result": output})  # fmt: skip

    record: dict[str, Any] = {
        "schema": 1, "run_id": run_id, "mode": "imported",
        "case_id": case.id if case else f"lf-{trace_id[:12]}",
        "split": case.split if case else None,
        "category": case.category if case else "imported",
        "question": question, "git_commit": None,
        "model": next((c.get("model") for c in chats if c.get("model")), None),
        "config_hash": None, "prompt_hash": fingerprint(system) if system else None,
        "recorded_at": root.get("startTime"), "today": (root.get("startTime") or "")[:10] or None,
        "wall_seconds": root.get("latency"), "llm_calls": llm_calls, "tool_calls": tool_calls,
        "source": {"langfuse_trace_id": trace_id, "root": root.get("name"),
                   "session_id": root.get("sessionId") or _attr(root, "langfuse.session.id"),
                   "tags": root.get("tags") or []},
        "failures": [],
    }  # fmt: skip
    answer = _json((agent or {}).get("output")) or _json(root.get("output"))
    failed = (agent or root).get("level") == "ERROR" or root.get("level") == "ERROR"
    if failed or not isinstance(answer, dict) or "claims" not in answer:
        message = (agent or root).get("statusMessage") or root.get("statusMessage")
        record.update(status="failed", error=message or "no answer in the trace", answer=None,
                      stats=None, failures=[f"run failed: {message or 'no answer in the trace'}"])
        return record, None
    answer["works"] = [_work(work_blocks[w]) for w in answer.get("references", [])
                       if w in work_blocks]  # fmt: skip
    answer["data"] = [{"id": r, "text": data_texts[r]} for r in answer.get("data_references", [])
                      if r in data_texts]  # fmt: skip
    stats = RunStats(
        iterations=int(_attr(agent or {}, "research_agent.iterations", len(chats))),
        search_calls=int(_attr(agent or {}, "research_agent.search_calls", 0)),
        input_tokens=int(_attr(agent or {}, "research_agent.input_tokens", 0)),
        output_tokens=int(_attr(agent or {}, "research_agent.output_tokens", 0)),
        works_shown=int(_attr(agent or {}, "research_agent.works_shown", 0)),
        cached_searches=int(_attr(agent or {}, "research_agent.cached_searches", 0)),
        rainfall_calls=int(_attr(agent or {}, "research_agent.rainfall_calls", 0)),
        cached_rainfall=int(_attr(agent or {}, "research_agent.cached_rainfall", 0)),
    )
    stats_data = asdict(stats)
    stats_data.update(tool_trace=trace, total_tokens=stats.total_tokens)
    record.update(status="ok", error=None, answer=answer, stats=stats_data)
    if root.get("name") in TURN_ROOTS:
        number = _attr(root, "research_agent.turn")
        record["conversation"] = {
            "id": case.turn.conversation if case and case.turn
            else record["source"]["session_id"],
            "turn": case.turn.turn if case and case.turn else number,
            "relation": _attr(root, "research_agent.relation"),
            "relation_source": _attr(root, "research_agent.relation_source"),
            "context_tokens": _attr(root, "research_agent.context.tokens", 0),
            "context_turns": _attr(root, "research_agent.context.turns_sent", 0),
            "context_omitted": _attr(root, "research_agent.context.omitted", 0),
            "recalled": recalled,
            "state": None,  # the task state is not in the trace
        }  # fmt: skip
    return record, None


def case_matcher(cases: list[EvalCase]) -> Callable[[dict[str, Any], str], EvalCase | None]:
    """An eval case by its tag; a conversation turn by question and turn number; else None."""
    by_id = {c.id: c for c in cases}
    turns: dict[tuple[str, int], EvalCase] = {}
    for c in cases:
        if c.turn:
            turns.setdefault((c.question, c.turn.turn), c)

    def match(root: dict[str, Any], question: str) -> EvalCase | None:
        if root.get("name") == "eval.case":
            for tag in root.get("tags") or []:
                if tag != "eval" and tag in by_id:
                    return by_id[tag]
            return None
        number = _attr(root, "research_agent.turn")
        if root.get("name") in TURN_ROOTS and number is not None:
            return turns.get((question.strip(), int(number)))
        return None

    return match
