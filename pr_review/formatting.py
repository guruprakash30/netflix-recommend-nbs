from __future__ import annotations

import base64
import json
import re

from pr_review.models import PRReviewState


_STATE_PATTERN = re.compile(r"<!-- ai-state:([A-Za-z0-9+/=]+) -->")

_VERDICT_ICON = {
    "lgtm": "✅",
    "needs_changes": "🔧",
    "security_issue": "🚨",
}

_SEVERITY_ICON = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}


def encode_state(state: PRReviewState) -> str:
    payload = {
        "summary": state.summary,
        "suggestions": state.suggestions,
        "security_issues": [si.model_dump() for si in state.security_issues],
        "overall_verdict": state.overall_verdict,
        "chat_history": state.chat_history,
        "file_changes": [fc.model_dump() for fc in state.file_changes],
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def decode_state(comment_body: str) -> dict | None:
    match = _STATE_PATTERN.search(comment_body)
    if not match:
        return None
    try:
        return json.loads(base64.b64decode(match.group(1).encode()).decode())
    except Exception:
        return None


def build_review_comment(state: PRReviewState) -> str:
    icon = _VERDICT_ICON.get(state.overall_verdict, "🔧")
    lines: list[str] = [
        "<!-- ai-review-bot -->",
        f"## {icon} AI Code Review",
        "",
        f"**Summary:** {state.summary}",
        "",
    ]

    if state.security_issues:
        lines += ["### 🚨 Security Issues", ""]
        for si in state.security_issues:
            sev_icon = _SEVERITY_ICON.get(si.severity, "⚪")
            lines.append(f"- {sev_icon} **{si.severity.upper()}**: {si.description}")
        lines.append("")

    if state.suggestions:
        lines += ["### 💡 Suggestions", ""]
        for s in state.suggestions:
            lines.append(f"- {s}")
        lines.append("")

    if state.file_changes:
        lines += ["### 📝 Proposed file changes", ""]
        for fc in state.file_changes:
            lines.append("<details>")
            lines.append(f"<summary><code>{fc.path}</code> — {fc.description}</summary>")
            lines.append("</details>")
            lines.append("")

    partial = [cs for cs in state.context_statuses if not cs.full_context]
    if partial:
        lines += ["### ⚠️ Partial analysis", ""]
        lines.append(
            "The following files were analysed with incomplete context — "
            "related files could not be fetched within the token budget. "
            "Suggestions may be incomplete; file changes were not proposed for these files."
        )
        lines.append("")
        for cs in partial:
            dropped = ", ".join(f"`{f}`" for f in cs.dropped_related)
            lines.append("<details>")
            lines.append(f"<summary><code>{cs.filepath}</code> — context incomplete</summary>")
            lines.append("")
            lines.append(f"**Not fetched:** {dropped}  ")
            lines.append(f"**Tokens used:** {cs.token_count:,}")
            lines.append("")
            lines.append("</details>")
        lines.append("")

    if state.chat_history:
        lines += ["### 💬 Review Discussion", ""]
        for msg in state.chat_history:
            role_label = "👤 Human" if msg["role"] == "human" else "🤖 AI"
            lines.append(f"**{role_label}:** {msg['content']}")
        lines.append("")

    lines += [
        "---",
        "_To interact: reply to this PR._",
        "- Reply **`approve`** → push AI-suggested changes as a new commit (you merge manually)",
        "- Reply **`reject`** → close review, no changes",
        "- Reply anything else → refine the review via chat",
        "",
        f"<!-- ai-state:{encode_state(state)} -->",
    ]

    return "\n".join(lines)


def build_lgtm_comment() -> str:
    return (
        "<!-- ai-review-bot -->\n"
        "✅ **AI Review: LGTM!**\n\n"
        "No issues found. Auto-merging."
    )


def build_reject_comment() -> str:
    return (
        "<!-- ai-review-bot -->\n"
        "❌ Review rejected. No changes applied."
    )


def build_commit_pushed_comment() -> str:
    return (
        "<!-- ai-review-bot -->\n"
        "🤖 **Changes pushed as a new commit.**\n\n"
        "AI-suggested edits have been added to this branch. "
        "Please review the commit and merge manually when ready."
    )
