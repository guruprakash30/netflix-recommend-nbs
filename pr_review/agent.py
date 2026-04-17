from __future__ import annotations

import os

from langgraph.graph import END, StateGraph

from pr_review import nodes, tools
from pr_review.formatting import build_reject_comment
from pr_review.models import FileChange, PRReviewState, SecurityIssue


def route_after_analysis(state: PRReviewState) -> str:
    if state.overall_verdict == "security_issue":
        return "handle_security"
    if state.overall_verdict == "lgtm":
        return "auto_merge"
    return "post_review"


def route_hitl(state: PRReviewState) -> str:
    if state.human_decision == "approve":
        return "commit_changes"
    return "process_human_input"


def build_graph() -> StateGraph:
    builder = StateGraph(PRReviewState)

    builder.add_node("fetch_diff",         nodes.fetch_diff_node)
    builder.add_node("extract_diff_files", nodes.extract_diff_files_node)
    builder.add_node("analyse_files",      nodes.analyse_files_node)
    builder.add_node("aggregate_results",  nodes.aggregate_results_node)
    builder.add_node("handle_security",    nodes.handle_security_node)
    builder.add_node("auto_merge",         nodes.auto_merge_node)
    builder.add_node("post_review",        nodes.post_review_node)

    builder.set_entry_point("fetch_diff")
    builder.add_edge("fetch_diff",         "extract_diff_files")
    builder.add_edge("extract_diff_files", "analyse_files")
    builder.add_edge("analyse_files",      "aggregate_results")

    builder.add_conditional_edges(
        "aggregate_results",
        route_after_analysis,
        {
            "handle_security": "handle_security",
            "auto_merge":      "auto_merge",
            "post_review":     "post_review",
        },
    )

    builder.add_edge("handle_security", "post_review")
    builder.add_edge("auto_merge",      END)
    builder.add_edge("post_review",     END)

    return builder.compile()


def build_hitl_graph() -> StateGraph:
    builder = StateGraph(PRReviewState)

    builder.add_node("hitl_router",         nodes.hitl_router_node)
    builder.add_node("process_human_input", nodes.process_human_input_node)
    builder.add_node("commit_changes",      nodes.commit_changes_node)

    builder.set_entry_point("hitl_router")
    builder.add_conditional_edges(
        "hitl_router",
        route_hitl,
        {
            "commit_changes":      "commit_changes",
            "process_human_input": "process_human_input",
        },
    )
    builder.add_edge("commit_changes",      END)
    builder.add_edge("process_human_input", END)

    return builder.compile()


def main() -> None:
    pr_number = int(os.environ["PR_NUMBER"])
    repo = os.environ["REPO_FULL_NAME"]
    event = os.environ.get("EVENT_NAME", "pull_request")

    if event == "pull_request":
        graph = build_graph()
        graph.invoke(PRReviewState(pr_number=pr_number, repo=repo))

    elif event == "issue_comment":
        comment_body = os.environ.get("COMMENT_BODY", "").strip()
        decision = (
            "approve" if comment_body.lower() == "approve"
            else "reject" if comment_body.lower() == "reject"
            else "chat"
        )

        prior_state, comment_id = tools.recover_prior_state(repo, pr_number)
        if prior_state is None:
            print("No prior AI review comment found — nothing to resume.")
            return

        pr_data = tools.get_pr_diff(repo, pr_number)

        if decision == "reject":
            tools.post_review_comment(repo, pr_number, build_reject_comment())
            return

        state = PRReviewState(
            pr_number=pr_number,
            repo=repo,
            diff=pr_data["diff"],
            pr_title=pr_data["title"],
            pr_body=pr_data["body"],
            summary=prior_state.get("summary", ""),
            suggestions=prior_state.get("suggestions", []),
            security_issues=[
                SecurityIssue(**si) for si in prior_state.get("security_issues", [])
            ],
            overall_verdict=prior_state.get("overall_verdict", "needs_changes"),
            chat_history=prior_state.get("chat_history", []),
            file_changes=[
                FileChange(**fc) for fc in prior_state.get("file_changes", [])
            ],
            review_comment_id=comment_id,
            hitl_active=True,
            human_decision=decision,
            human_message=comment_body,
        )

        build_hitl_graph().invoke(state)


if __name__ == "__main__":
    main()
