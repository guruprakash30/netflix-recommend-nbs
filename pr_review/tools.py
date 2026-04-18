from __future__ import annotations

import base64
import os
import tempfile

import patch as patch_lib
import requests

from pr_review.formatting import decode_state


GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE = "https://api.github.com"


def get_pr_diff(repo: str, pr_number: int) -> dict:
    pr_url = f"{BASE}/repos/{repo}/pulls/{pr_number}"
    pr = requests.get(pr_url, headers=HEADERS).json()
    diff_headers = {**HEADERS, "Accept": "application/vnd.github.v3.diff"}
    diff = requests.get(pr_url, headers=diff_headers).text
    return {
        "title": pr["title"],
        "body": pr.get("body") or "",
        "diff": diff,
        "base_branch": pr["base"]["ref"],
        "head_branch": pr["head"]["ref"],
        "author": pr["user"]["login"],
    }


def post_review_comment(repo: str, pr_number: int, body: str) -> int:
    url = f"{BASE}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=HEADERS, json={"body": body})
    resp.raise_for_status()
    return resp.json()["id"]


def update_comment(repo: str, comment_id: int, body: str) -> None:
    url = f"{BASE}/repos/{repo}/issues/comments/{comment_id}"
    requests.patch(url, headers=HEADERS, json={"body": body})


def get_bot_comment(repo: str, pr_number: int) -> dict | None:
    url = f"{BASE}/repos/{repo}/issues/{pr_number}/comments"
    comments = requests.get(url, headers=HEADERS, params={"per_page": 100}).json()
    for comment in reversed(comments):
        if "<!-- ai-review-bot -->" in comment.get("body", ""):
            return {"id": comment["id"], "body": comment["body"]}
    return None

def delete_all_bot_comments(repo: str, pr_number: int) -> None:
    url = f"{BASE}/repos/{repo}/issues/{pr_number}/comments"
    comments = requests.get(url, headers=HEADERS, params={"per_page": 100}).json()
    for comment in comments:
        if "<!-- ai-review-bot -->" in comment.get("body", ""):
            requests.delete(
                f"{BASE}/repos/{repo}/issues/comments/{comment['id']}",
                headers=HEADERS,
            )

def recover_prior_state(repo: str, pr_number: int) -> tuple[dict | None, int | None]:
    bot_comment = get_bot_comment(repo, pr_number)
    if not bot_comment:
        return None, None
    return decode_state(bot_comment["body"]), bot_comment["id"]


LABEL_COLORS = {
    "security-high": "d73a4a",
    "security-medium": "e4e669",
    "security-low": "0075ca",
}

def get_pr_security_labels(repo: str, pr_number: int) -> list[str]:
    resp = requests.get(
        f"{BASE}/repos/{repo}/issues/{pr_number}/labels",
        headers=HEADERS,
    )
    return [
        l["name"] for l in resp.json()
        if l["name"].startswith("security-")
    ]


def apply_label(repo: str, pr_number: int, label: str) -> None:
    requests.post(
        f"{BASE}/repos/{repo}/labels",
        headers=HEADERS,
        json={"name": label, "color": LABEL_COLORS.get(label, "ededed")},
    )
    requests.post(
        f"{BASE}/repos/{repo}/issues/{pr_number}/labels",
        headers=HEADERS,
        json={"labels": [label]},
    )


def remove_label(repo: str, pr_number: int, label: str) -> None:
    requests.delete(
        f"{BASE}/repos/{repo}/issues/{pr_number}/labels/{label}",
        headers=HEADERS,
    )


def merge_pr(repo: str, pr_number: int, commit_title: str) -> None:
    requests.put(
        f"{BASE}/repos/{repo}/pulls/{pr_number}/merge",
        headers=HEADERS,
        json={"commit_title": f"Auto-merged: {commit_title}", "merge_method": "squash"},
    )


def _apply_patch_to_content(filepath: str, content: str, patch_text: str) -> str | None:
    basename = os.path.basename(filepath)
    full_patch = f"--- a/{basename}\n+++ b/{basename}\n{patch_text}"

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, basename)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        pset = patch_lib.fromstring(full_patch.encode())
        if not pset.apply(root=tmpdir):
            return None

        with open(target, "r", encoding="utf-8") as f:
            return f.read()


def push_commit_to_pr(repo: str, pr_number: int, file_changes: list[dict], commit_msg: str) -> None:
    pr = requests.get(f"{BASE}/repos/{repo}/pulls/{pr_number}", headers=HEADERS).json()
    branch = pr["head"]["ref"]

    for change in file_changes:
        file_resp = requests.get(
            f"{BASE}/repos/{repo}/contents/{change['path']}",
            headers=HEADERS,
            params={"ref": branch},
        )
        if file_resp.status_code != 200:
            continue

        file_data = file_resp.json()
        current_content = base64.b64decode(file_data["content"]).decode("utf-8")
        sha = file_data["sha"]

        new_content = _apply_patch_to_content(change["path"], current_content, change["patch"])
        if new_content is None:
            continue

        requests.put(
            f"{BASE}/repos/{repo}/contents/{change['path']}",
            headers=HEADERS,
            json={
                "message": commit_msg,
                "content": base64.b64encode(new_content.encode()).decode(),
                "branch": branch,
                "sha": sha,
            },
        )
