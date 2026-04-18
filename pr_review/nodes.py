from __future__ import annotations
import os
from langchain_openai import AzureChatOpenAI


from langchain_core.messages import HumanMessage, SystemMessage

from pr_review.file_analysis_agent import analyse_file
from pr_review.formatting import (
    build_commit_pushed_comment,
    build_lgtm_comment,
    build_review_comment,
)
from pr_review.helpers import _extract_diff_files, _fetch_primary_file
from pr_review.models import (
    LLMAggregateResult,
    LLMRefinedResult,
    PerFileAnalysis,
    PRReviewState,
)
from pr_review import tools


base_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4.1-mini",
    api_version="2025-01-01-preview",
    temperature=0,
)
refine_llm = base_llm.with_structured_output(LLMRefinedResult)
_aggregate_llm = base_llm.with_structured_output(LLMAggregateResult)


REFINE_SYSTEM = """You are a code review assistant helping a human refine a PR review.

You will receive:
- The original git diff
- Your previous suggestions
- The full conversation so far

Your job:
1. Acknowledge the human's input in ai_reply (natural language, concise)
2. Revise summary / suggestions / security_issues / overall_verdict as needed
3. If the human asked for specific code changes, populate file_changes with patch-format
   changes for each affected file. If no code changes are needed, leave it empty.
"""

AGGREGATE_SYSTEM = """You are a senior engineer writing the final summary of an automated PR review.

You will receive per-file analysis results. Your job:
1. Write a 3-5 sentence PR-level summary — synthesise across files, do NOT repeat
   the same point more than once even if it appeared in multiple file analyses.
   If two files have the same class of issue, mention it once with both files named.
2. Merge suggestions — remove duplicates and semantically equivalent items.
   Keep the clearest wording, drop the rest.
3. Merge security_issues — deduplicate by description similarity.
4. overall_verdict = most severe verdict across all files
   (security_issue > needs_changes > lgtm).
"""


def fetch_diff_node(state: PRReviewState) -> dict:
    pr_data = tools.get_pr_diff(state.repo, state.pr_number)
    return {
        "diff": pr_data["diff"],
        "pr_title": pr_data["title"],
        "pr_body": pr_data["body"],
        "head_branch": pr_data["head_branch"],
    }


def extract_diff_files_node(state: PRReviewState) -> dict:
    file_hunks = _extract_diff_files(state.diff)
    return {
        "diff_file_hunks": [
            {"filepath": fp, "hunk": hunk} for fp, hunk in file_hunks
        ]
    }


def analyse_files_node(state: PRReviewState) -> dict:
    analyses: list[PerFileAnalysis] = []

    for entry in state.diff_file_hunks:
        filepath = entry["filepath"]
        hunk = entry["hunk"]

        primary_content = _fetch_primary_file(state.repo, state.head_branch, filepath)
        if primary_content is None:
            continue

        analysis = analyse_file(
            repo=state.repo,
            branch=state.head_branch,
            filepath=filepath,
            diff_hunk=hunk,
            primary_content=primary_content,
        )
        analyses.append(analysis)

    return {"per_file_analyses": analyses}


def aggregate_results_node(state: PRReviewState) -> dict:
    if not state.per_file_analyses:
        return {
            "summary": "No files could be analysed.",
            "overall_verdict": "needs_changes",
            "suggestions": [],
            "security_issues": [],
            "file_changes": [],
            "context_statuses": [],
        }

    file_blocks = []
    for a in state.per_file_analyses:
        partial_warn = (
            f"  ⚠️ PARTIAL CONTEXT — dropped: {', '.join(a.context_status.dropped_related)}\n"
            if not a.context_status.full_context
            else ""
        )
        suggestions_text = "\n".join(f"  - {s}" for s in a.suggestions) or "  (none)"
        security_text = (
            "\n".join(f"  - [{si.severity}] {si.description}" for si in a.security_issues)
            or "  (none)"
        )
        file_blocks.append(
            f"### {a.filepath} [{a.overall_verdict}]\n"
            f"{partial_warn}"
            f"Suggestions:\n{suggestions_text}\n"
            f"Security:\n{security_text}\n"
        )

    prompt = "\n".join(file_blocks)

    result: LLMAggregateResult = _aggregate_llm.invoke([
        SystemMessage(content=AGGREGATE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    safe_file_changes = [
        fc
        for a in state.per_file_analyses
        if a.context_status.full_context
        for fc in a.file_changes
    ]

    return {
        "summary": result.summary,
        "overall_verdict": result.overall_verdict,
        "suggestions": result.suggestions,
        "security_issues": result.security_issues,
        "file_changes": safe_file_changes,
        "context_statuses": [a.context_status for a in state.per_file_analyses],
    }


def handle_security_node(state: PRReviewState) -> None:
    existing = tools.get_pr_security_labels(state.repo, state.pr_number)
    for label in existing:
        tools.remove_label(state.repo, state.pr_number, label)
    for issue in state.security_issues:
        label = f"security-{issue.severity}"
        tools.apply_label(state.repo, state.pr_number, label)


def post_review_node(state: PRReviewState) -> dict:
    body = build_review_comment(state)
    existing = tools.get_bot_comment(state.repo, state.pr_number)
    if existing:
        tools.update_comment(state.repo, existing["id"], body)
        return {"review_comment_id": existing["id"], "hitl_active": True}
    comment_id = tools.post_review_comment(state.repo, state.pr_number, body)
    return {"review_comment_id": comment_id, "hitl_active": True}


def auto_merge_node(state: PRReviewState) -> dict:
    tools.post_review_comment(state.repo, state.pr_number, build_lgtm_comment())
    tools.merge_pr(state.repo, state.pr_number, state.pr_title)
    return {}


def hitl_router_node(state: PRReviewState) -> dict:
    return {}


def process_human_input_node(state: PRReviewState) -> dict:
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state.chat_history
    )

    prior_changes_text = (
        "\n".join(
            f"- {fc.path}: {fc.description or '(no description)'}"
            for fc in state.file_changes
        )
        if state.file_changes
        else "None"
    )

    prompt = (
        f"Original diff (truncated to 8k chars):\n{state.diff[:8000]}\n\n"
        f"Previous suggestions:\n"
        + "\n".join(f"- {s}" for s in state.suggestions)
        + f"\n\nFile changes already proposed (path + description only):\n{prior_changes_text}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Human's latest message: {state.human_message}"
    )

    result: LLMRefinedResult = refine_llm.invoke([
        SystemMessage(content=REFINE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    new_messages = [
        {"role": "human", "content": state.human_message},
        {"role": "ai", "content": result.ai_reply},
    ]

    updated_fields = {
        "summary": result.summary,
        "overall_verdict": result.overall_verdict,
        "suggestions": result.suggestions,
        "security_issues": result.security_issues,
        "file_changes": result.file_changes,
        "chat_history": new_messages,
    }

    if state.review_comment_id:
        updated_state = state.model_copy(update={
            **updated_fields,
            "chat_history": state.chat_history + new_messages,
        })
        tools.update_comment(
            state.repo,
            state.review_comment_id,
            build_review_comment(updated_state),
        )

    return updated_fields


def commit_changes_node(state: PRReviewState) -> dict:
    if state.file_changes:
        tools.push_commit_to_pr(
            state.repo,
            state.pr_number,
            [fc.model_dump() for fc in state.file_changes],
            "chore: apply AI review suggestions",
        )
    tools.post_review_comment(
        state.repo,
        state.pr_number,
        build_commit_pushed_comment(),
    )
    return {}
