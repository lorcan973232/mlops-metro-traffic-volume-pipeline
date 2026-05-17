from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPORT_PATH = Path("reports/submission/public_repository_evidence.json")
PUBLIC_UNTIL = "2026-06-21"


# ==============================================================================
# Public repository evidence
# ==============================================================================
#
# The assignment requires a public personal GitHub repository until 21 June 2026.
# This script can only prove current visibility, so it records a timestamped
# snapshot and an explicit future-responsibility note instead of making a false
# permanent claim.


def run_command(args: list[str]) -> tuple[int, str, str]:
    """Run GitHub or git commands and return a structured failure if blocked."""
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def current_sha() -> str:
    """Return the commit SHA that the visibility evidence belongs to."""
    code, stdout, _ = run_command(["git", "rev-parse", "HEAD"])
    return stdout if code == 0 and stdout else "unknown"


def detect_repository() -> tuple[str, str]:
    """Detect the GitHub repository from Actions context or the local remote."""
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if repository:
        return repository, f"https://github.com/{repository}"

    code, stdout, _ = run_command(["git", "remote", "get-url", "origin"])
    if code != 0 or not stdout:
        raise SystemExit(
            "Repository URL could not be detected from GITHUB_REPOSITORY or git remote."
        )

    url = stdout
    if url.startswith("git@github.com:"):
        repository = url.removeprefix("git@github.com:").removesuffix(".git")
        return repository, f"https://github.com/{repository}"
    if "github.com/" in url:
        repository = url.split("github.com/", 1)[1].removesuffix(".git").strip("/")
        return repository, f"https://github.com/{repository}"
    raise SystemExit(f"Unsupported repository remote URL for GitHub visibility check: {url}")


def gh_visibility(repository: str) -> tuple[dict[str, Any] | None, str]:
    """Prefer GitHub CLI visibility because it matches the student's account view."""
    fields = "nameWithOwner,url,visibility,isPrivate,defaultBranchRef"
    code, stdout, stderr = run_command(["gh", "repo", "view", repository, "--json", fields])
    if code != 0 or not stdout:
        return None, f"gh repo view failed or unavailable: {stderr or 'no output'}"
    payload = json.loads(stdout)
    return {
        "repository_url": payload["url"],
        "full_name": payload["nameWithOwner"],
        "visibility": str(payload["visibility"]).lower(),
        "private": bool(payload["isPrivate"]),
        "default_branch": payload.get("defaultBranchRef", {}).get("name"),
    }, "gh repo view --json nameWithOwner,url,visibility,isPrivate,defaultBranchRef"


def api_visibility(repository: str, repository_url: str) -> tuple[dict[str, Any] | None, str]:
    """Fallback to the GitHub REST API for workflow or unauthenticated checks."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "mlops-artefact-visibility-check",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"https://api.github.com/repos/{repository}", headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        return None, f"GitHub REST API check failed: {exc}"

    return {
        "repository_url": payload.get("html_url", repository_url),
        "full_name": payload["full_name"],
        "visibility": payload.get("visibility"),
        "private": bool(payload["private"]),
        "default_branch": payload.get("default_branch"),
    }, "GitHub REST API /repos/{owner}/{repo}"


def build_evidence() -> dict[str, Any]:
    """Build the public-visibility evidence JSON consumed by README and workflows."""
    repository, repository_url = detect_repository()
    payload, method = gh_visibility(repository)
    fallback_note = None
    if payload is None:
        fallback_note = method
        payload, method = api_visibility(repository, repository_url)
    if payload is None:
        raise SystemExit(f"Repository visibility could not be verified. {method}")

    evidence = {
        **payload,
        "evidence_scope": (
            "Snapshot generated at checked_at. The latest workflow artefact is the "
            "authoritative current visibility proof for the submitted commit."
        ),
        "checked_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "latest_commit_sha_at_check_time": current_sha(),
        "command_or_method_used": method,
        "fallback_note": fallback_note,
        "requirement": "Repository must remain public until 21 June 2026.",
        "requirement_note": (
            "Current public visibility is verified at checked_at. The repository must remain "
            "public until 21 June 2026 for assignment compliance."
        ),
        "public_until_required_date": PUBLIC_UNTIL,
        "current_public_visibility_verified": (
            payload.get("private") is False and str(payload.get("visibility")).lower() == "public"
        ),
        "future_public_until_proven": False,
        "future_compliance_note": (
            "Current public visibility is verified. The future public-until date is safeguarded "
            "through documentation and a repeatable visibility-check workflow, but the student "
            "must keep the repository public until 21 June 2026."
        ),
    }
    return evidence


def main() -> None:
    """Write visibility evidence and fail if the repository is not public now."""
    evidence = build_evidence()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if not evidence["current_public_visibility_verified"]:
        raise SystemExit(f"Repository is not public: {evidence}")


if __name__ == "__main__":
    main()
