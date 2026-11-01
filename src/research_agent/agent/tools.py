"""The ``search_literature`` tool: schema, argument validation and result formatting.

Formatting is deterministic and enforces the retrieved-context budget, so the model never sees
more than ``max_context_tokens`` of retrieved text across a whole question.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from research_agent.agent.budget import estimate_tokens
from research_agent.llm import ToolSpec
from research_agent.retrieval import SearchResult, Work

SEARCH_TOOL_NAME = "search_literature"

SEARCH_TOOL = ToolSpec(
    name=SEARCH_TOOL_NAME,
    description=(
        "Search scholarly literature (OpenAlex) by title and abstract. Returns titles, authors, "
        "year, venue and abstracts, each with a work ID such as W4388333579. Use short keyword "
        "queries. Results are matched on words in the title or abstract."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short keyword query."},
            "year_from": {
                "type": "integer",
                "description": "Optional. Only works published in or after this year.",
            },
            "max_results": {
                "type": "integer",
                "description": "Optional. Number of results wanted; capped by the system.",
            },
        },
        "required": ["query"],
    },
)


class ToolArgumentError(ValueError):
    """The model supplied unusable tool arguments. The message is safe to show the model."""


@dataclass(frozen=True)
class SearchArgs:
    query: str
    year_from: int | None
    max_results: int | None


def _optional_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolArgumentError(f"'{key}' must be an integer")
    return value


def parse_search_args(arguments: dict[str, Any]) -> SearchArgs:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolArgumentError("'query' is required and must be a non-empty string")
    return SearchArgs(
        query=query,
        year_from=_optional_int(arguments, "year_from"),
        max_results=_optional_int(arguments, "max_results"),
    )


def _format_work(work: Work) -> str:
    authors = ", ".join(work.authors[:3]) + (" et al." if len(work.authors) > 3 else "")
    meta = ", ".join(
        part
        for part in (
            authors,
            work.venue,
            str(work.year) if work.year else None,
            f"cited by {work.cited_by_count}" if work.cited_by_count is not None else None,
        )
        if part
    )
    head = f"[{work.id}] {work.title or '(no title)'}" + (f" ({meta})" if meta else "")
    return f"{head}\nAbstract: {work.abstract or '(not available)'}"


@dataclass(frozen=True)
class FormattedResults:
    text: str
    shown: tuple[Work, ...]  # works whose full record the model saw (citable evidence)
    omitted: int  # results left out because the context budget ran out
    context_tokens: int  # estimated tokens of retrieved content in ``text``


def format_results(
    result: SearchResult, context_left: int, already_seen: Collection[str]
) -> FormattedResults:
    """Render results for the model within ``context_left`` estimated tokens.

    Results keep OpenAlex's relevance order; once one does not fit, it and all later ones are
    omitted. Works already shown in an earlier search are listed by ID only.
    """
    lines = [f'Search "{result.query}": showing results of {result.total_count} matches.']
    used = estimate_tokens(lines[0])
    shown: list[Work] = []
    omitted = 0
    for index, work in enumerate(result.works):
        block = (
            f"[{work.id}] (already shown in an earlier search)"
            if work.id in already_seen
            else _format_work(work)
        )
        cost = estimate_tokens(block)
        if used + cost > context_left:
            omitted = len(result.works) - index
            break
        lines.append(block)
        used += cost
        if work.id not in already_seen:
            shown.append(work)
    if not result.works:
        lines.append("No results.")
    if omitted:
        lines.append(f"({omitted} more result(s) omitted: retrieved-context budget reached.)")
    return FormattedResults(
        text="\n\n".join(lines), shown=tuple(shown), omitted=omitted, context_tokens=used
    )
