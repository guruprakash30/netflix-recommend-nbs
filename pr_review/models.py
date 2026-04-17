from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SecurityIssue(BaseModel):
    description: str = Field(
        description="Clear description of the security issue found."
    )
    severity: Literal["high", "medium", "low"] = Field(
        description=(
            "high = actively exploitable or exposes sensitive data; "
            "medium = significant risk requiring non-trivial conditions to exploit; "
            "low = minor concern, defence-in-depth improvement."
        )
    )


class FileChange(BaseModel):
    path: str = Field(
        description=(
            "Relative file path from repo root, exactly as it appears in the diff header "
            "(e.g. 'src/auth/login.py'). Must be a file whose FULL content you received — "
            "never propose changes to a file you only partially saw."
        )
    )
    patch: str = Field(
        description=(
            "A unified diff patch string describing only your changes. "
            "Format exactly like a git diff hunk:\n"
            "  - @@ -old_start,old_count +new_start,new_count @@ on the first line\n"
            "  - 3 lines of unchanged context before and after your change\n"
            "  - lines to remove prefixed with -\n"
            "  - lines to add prefixed with +\n"
            "Every context line must match the actual file content you were given exactly. "
            "Do not invent or paraphrase context lines — patch application will fail if they don't match. "
            "If you are uncertain whether your context lines are exact, omit this file from file_changes "
            "and add a suggestion instead."
        )
    )
    description: str = Field(
        description=(
            "Single-line plain-English summary shown to the reviewer in the PR comment, "
            "e.g. 'Fix SQL injection on line 13 — use parameterized query instead of f-string'. "
            "Be specific about what changed and why. This is all the reviewer sees before approving."
        )
    )


class ContextStatus(BaseModel):
    filepath: str = Field(description="The primary diff file this status refers to.")
    full_context: bool = Field(
        description=(
            "True if the primary file AND all related files were fully fetched "
            "within the token budget. False if any related file was dropped or truncated."
        )
    )
    fetched_related: list[str] = Field(
        default_factory=list,
        description="Related files that were successfully fetched and included."
    )
    dropped_related: list[str] = Field(
        default_factory=list,
        description=(
            "Related files the LLM identified as relevant but which were dropped "
            "because the token budget was reached. Empty when full_context is True."
        )
    )
    token_count: int = Field(
        default=0,
        description="Approximate token count of the context window used for this file's invoke call."
    )


class PerFileAnalysis(BaseModel):
    filepath: str
    context_status: ContextStatus
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Actionable suggestions for this file. Safe to populate regardless of context completeness — "
            "but if full_context is False, prefix each suggestion with [PARTIAL CONTEXT]."
        )
    )
    security_issues: list[SecurityIssue] = Field(
        default_factory=list,
        description=(
            "Security issues found in this file. "
            "Only populate when you have seen enough context to be confident in the finding. "
            "A false negative is safer than a false positive here."
        )
    )
    file_changes: list[FileChange] = Field(
        default_factory=list,
        description=(
            "Patch-format changes to propose as a commit. "
            "ONLY populate this when context_status.full_context is True. "
            "If full_context is False, leave this empty and put the fix in suggestions instead. "
            "Every FileChange.path must appear in context_status.fetched_related or equal filepath."
        )
    )
    overall_verdict: Literal["lgtm", "needs_changes", "security_issue"] = Field(
        description=(
            "Verdict for this file only — not the whole PR. "
            "lgtm = this file looks good given the context available; "
            "needs_changes = quality or correctness issues found; "
            "security_issue = any security issue found regardless of quality."
        )
    )


class LLMRefinedResult(BaseModel):
    summary: str = Field(description="Updated PR-level summary incorporating human feedback.")
    overall_verdict: Literal["lgtm", "needs_changes", "security_issue"]
    suggestions: list[str] = Field(default_factory=list)
    security_issues: list[SecurityIssue] = Field(default_factory=list)
    ai_reply: str = Field(
        description=(
            "Natural language reply to the human's message. "
            "If you are revising a suggestion that was marked partial/incomplete, "
            "acknowledge that the revision is still operating on incomplete context."
        )
    )
    file_changes: list[FileChange] = Field(
        default_factory=list,
        description=(
            "Only propose file changes for files where full_context was True in the original analysis. "
            "Check the context_statuses in the PR state before proposing."
        )
    )


class LLMAggregateResult(BaseModel):
    summary: str = Field(
        description=(
            "3-5 sentence PR-level summary synthesised across all files. "
            "Never repeat the same issue twice even if it appeared in multiple files — "
            "mention it once with all affected files named inline. "
            "If any files had partial context, note this briefly at the end."
        )
    )
    overall_verdict: Literal["lgtm", "needs_changes", "security_issue"] = Field(
        description="Most severe verdict across all files."
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Merged, deduplicated suggestions across all files. "
            "If two suggestions are semantically equivalent keep the clearest one. "
            "Prefix each with the affected filepath(s) in brackets, "
            "e.g. '[src/auth/login.py] Use parameterized queries to prevent SQL injection'."
        )
    )
    security_issues: list[SecurityIssue] = Field(
        default_factory=list,
        description="Merged, deduplicated security issues across all files."
    )


class PRReviewState(BaseModel):
    pr_number: int
    repo: str
    diff: str = ""
    pr_title: str = ""
    pr_body: str = ""
    head_branch: str = ""

    diff_file_hunks: list[dict] = Field(default_factory=list)
    per_file_analyses: list[PerFileAnalysis] = Field(default_factory=list)

    summary: str = ""
    suggestions: list[str] = Field(default_factory=list)
    security_issues: list[SecurityIssue] = Field(default_factory=list)
    overall_verdict: Literal["lgtm", "needs_changes", "security_issue"] = "needs_changes"
    file_changes: list[FileChange] = Field(default_factory=list)
    context_statuses: list[ContextStatus] = Field(default_factory=list)

    hitl_active: bool = False
    chat_history: Annotated[list[dict], operator.add] = Field(default_factory=list)
    human_decision: Literal["approve", "reject", "chat"] | None = None
    human_message: str = ""

    review_comment_id: int | None = None
    labels_applied: list[str] = Field(default_factory=list)
