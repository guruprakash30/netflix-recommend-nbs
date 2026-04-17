import os
from __future__ import annotations

import base64
import tiktoken
from dataclasses import dataclass, field

import requests
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

from pr_review import tools as github_tools
from pr_review.models import ContextStatus, PerFileAnalysis


MAX_CONTEXT_TOKENS = 20_000
MAX_RELATED_FILES = 5
_enc = tiktoken.encoding_for_model("gpt-4o")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class FileAnalysisContext:
    repo: str
    branch: str
    primary_filepath: str
    fetched_files: dict[str, str] = field(default_factory=dict)
    dropped_files: list[str] = field(default_factory=list)
    token_budget_remaining: int = MAX_CONTEXT_TOKENS
    budget_hit: bool = False


@tool
def fetch_file(filepath: str, runtime) -> str:
    """
    Fetch the full content of a file from the repository.
    Only use this for files that are part of this repository's own source code —
    never for third-party packages or standard library modules.
    filepath must be the exact relative path from the repo root.

    Args:
        filepath: Relative path from repo root, e.g. 'src/models/user.py'
    """
    ctx: FileAnalysisContext = runtime.context

    if filepath in ctx.fetched_files:
        return f"[already fetched: {filepath}]"

    if len(ctx.fetched_files) >= MAX_RELATED_FILES:
        ctx.dropped_files.append(filepath)
        ctx.budget_hit = True
        return (
            f"[LIMIT REACHED: cannot fetch '{filepath}'. "
            f"You have already fetched the maximum of {MAX_RELATED_FILES} related files "
            f"({', '.join(ctx.fetched_files.keys())}). "
            f"Do NOT call fetch_file again. "
            f"Produce your final PerFileAnalysis now with full_context=False.]"
        )

    resp = requests.get(
        f"{github_tools.BASE}/repos/{ctx.repo}/contents/{filepath}",
        headers=github_tools.HEADERS,
        params={"ref": ctx.branch},
    )
    if resp.status_code != 200:
        ctx.dropped_files.append(filepath)
        return f"[NOT FOUND: {filepath} does not exist on branch {ctx.branch}]"

    content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="replace")
    token_cost = _count_tokens(content)

    if token_cost > ctx.token_budget_remaining:
        ctx.dropped_files.append(filepath)
        ctx.budget_hit = True
        return (
            f"[BUDGET: token limit reached fetching {filepath} "
            f"({token_cost} tokens needed, {ctx.token_budget_remaining} remaining). "
            f"Do NOT attempt further fetches. Mark this file's context as partial.]"
        )

    ctx.fetched_files[filepath] = content
    ctx.token_budget_remaining -= token_cost
    return f"### {filepath}\n\n```\n{content}\n```"


class TokenBudgetMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        ctx: FileAnalysisContext | None = getattr(request.runtime, "context", None)
        if ctx and ctx.budget_hit:
            nudged = request.override(
                messages=[
                    *request.messages,
                    SystemMessage(
                        content=(
                            "Token budget exhausted. Do not call fetch_file again. "
                            "Produce your final PerFileAnalysis now with full_context=False."
                        )
                    ),
                ]
            )
            return handler(nudged)
        return handler(request)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        ctx: FileAnalysisContext | None = getattr(
            getattr(request, "runtime", None), "context", None
        )
        if ctx and ctx.budget_hit and request.tool_call["name"] == "fetch_file":
            filepath = request.tool_call["args"].get("filepath", "unknown")
            ctx.dropped_files.append(filepath)
            return ToolMessage(
                content=(
                    f"[BUDGET EXHAUSTED: {filepath} not fetched. "
                    "Conclude analysis with partial context.]"
                ),
                tool_call_id=request.tool_call["id"],
            )
        return handler(request)


def _build_system_prompt(filepath: str, diff_hunk: str, primary_content: str) -> str:
    return f"""You are an expert code reviewer analysing a single file from a pull request.

Primary file under review: {filepath}
Token budget for related files: {MAX_CONTEXT_TOKENS} tokens, max {MAX_RELATED_FILES} files.

## Your task
1. Read the diff hunk and the primary file content below.
2. Use fetch_file to retrieve related files that are necessary to understand the change.
   Only fetch files that are part of this repository's own source code —
   do NOT fetch third-party packages, standard library modules, or any installed dependencies
   (e.g. do not fetch requests, pydantic, langchain, os, re, or anything from site-packages).
   Good candidates: other modules in this repo that this file imports from, schema files,
   config files, callers of functions changed in the diff.
3. Stop fetching when you have enough context or when a tool response tells you to stop.
4. Produce a PerFileAnalysis as your final structured output.

## Rules for file_changes
- Only populate file_changes if you received full context (no budget or limit messages from tools).
- If any fetch was blocked, leave file_changes empty and put your recommended fix in suggestions instead.
- Every patch context line must exactly match the actual file content you received —
  do not paraphrase or reconstruct lines you did not see.

## Rules for suggestions
- Always populate suggestions regardless of context completeness.
- If context was partial, prefix the affected suggestion with [PARTIAL CONTEXT].

## Rules for security_issues
- Only report a security issue when you have seen enough context to be confident.
  A missed finding is safer than a false positive.

## Diff hunk for {filepath}
```diff
{diff_hunk}
```

## Primary file content
```
{primary_content}
```"""


def _make_file_analysis_agent():
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="gpt-4.1-mini",
        api_version="2025-01-01-preview",
        temperature=0,
        max_tokens=2000,
    )
    return create_agent(
        model=model,
        tools=[fetch_file],
        response_format=ToolStrategy(PerFileAnalysis),
        middleware=[TokenBudgetMiddleware()],
        name="file_analysis_agent",
        context_schema=FileAnalysisContext,
    )


_agent = _make_file_analysis_agent()


def analyse_file(
    repo: str,
    branch: str,
    filepath: str,
    diff_hunk: str,
    primary_content: str,
) -> PerFileAnalysis:
    ctx = FileAnalysisContext(
        repo=repo,
        branch=branch,
        primary_filepath=filepath,
        token_budget_remaining=MAX_CONTEXT_TOKENS - _count_tokens(primary_content),
    )

    result = _agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _build_system_prompt(filepath, diff_hunk, primary_content),
                }
            ]
        },
        context=ctx,
    )

    analysis: PerFileAnalysis = result["structured_response"]
    return analysis.model_copy(update={
        "context_status": ContextStatus(
            filepath=filepath,
            full_context=not ctx.budget_hit and len(ctx.dropped_files) == 0,
            fetched_related=list(ctx.fetched_files.keys()),
            dropped_related=ctx.dropped_files,
            token_count=MAX_CONTEXT_TOKENS - ctx.token_budget_remaining,
        )
    })
