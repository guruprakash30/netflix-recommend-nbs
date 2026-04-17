from __future__ import annotations

import base64
import re

import requests

from pr_review import tools as github_tools


def _extract_diff_files(diff: str) -> list[tuple[str, str]]:
    file_blocks: list[tuple[str, str]] = []
    raw_blocks = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
    for block in raw_blocks:
        if not block.strip():
            continue
        match = re.search(r"^\+\+\+ b/(.+)$", block, re.MULTILINE)
        if match:
            file_blocks.append((match.group(1), block))
    return file_blocks


def _fetch_primary_file(repo: str, branch: str, filepath: str) -> str | None:
    resp = requests.get(
        f"{github_tools.BASE}/repos/{repo}/contents/{filepath}",
        headers=github_tools.HEADERS,
        params={"ref": branch},
    )
    if resp.status_code != 200:
        return None
    return base64.b64decode(resp.json()["content"]).decode("utf-8", errors="replace")
