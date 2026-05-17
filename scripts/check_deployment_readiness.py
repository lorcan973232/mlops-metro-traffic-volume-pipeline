"""Verify Docker and Kind deployment readiness for the current artefact.

The preferred proof is a local Docker/Kind run when those tools are installed.
If the local machine lacks Docker, Kind, or kubectl, this script falls back to
the current GitHub Actions evidence for the exact commit SHA. That gives a
marker one command which separates a real artefact failure from a local setup
blocker.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Any

REQUIRED_WORKFLOWS = ("Docker Build", "Deploy Kind")


def run_command(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def current_sha() -> str:
    """Return the current Git commit SHA, or unknown if Git is unavailable."""
    code, stdout, _ = run_command(["git", "rev-parse", "HEAD"])
    return stdout if code == 0 and stdout else "unknown"


def local_tool_status() -> dict[str, Any]:
    """Check whether Docker, Kind, and kubectl are available locally."""
    tools: dict[str, Any] = {}
    for tool in ("docker", "kind", "kubectl"):
        path = shutil.which(tool)
        tools[tool] = {
            "available": path is not None,
            "path": path,
        }

    docker_daemon = {"checked": False, "available": False, "detail": "docker not on PATH"}
    if tools["docker"]["available"]:
        code, stdout, stderr = run_command(["docker", "info"], timeout=60)
        docker_daemon = {
            "checked": True,
            "available": code == 0,
            "detail": stdout[:500] if code == 0 else stderr[:500],
        }

    return {
        "tools": tools,
        "docker_daemon": docker_daemon,
        "all_local_deployment_tools_ready": (
            all(item["available"] for item in tools.values()) and docker_daemon["available"]
        ),
    }


def github_workflow_status(sha: str) -> dict[str, Any]:
    """Read current-SHA Docker/Kind workflow status with GitHub CLI."""
    if not shutil.which("gh"):
        return {
            "available": False,
            "reason": "gh is not installed or not on PATH",
            "current_sha": sha,
            "workflow_runs": {},
            "missing_workflows": list(REQUIRED_WORKFLOWS),
            "unsuccessful_workflows": [],
        }

    code, stdout, stderr = run_command(
        [
            "gh",
            "run",
            "list",
            "--branch",
            "main",
            "--limit",
            "40",
            "--json",
            "headSha,name,status,conclusion,url,createdAt,event",
        ],
        timeout=120,
    )
    if code != 0:
        return {
            "available": False,
            "reason": stderr or stdout or "gh run list failed",
            "current_sha": sha,
            "workflow_runs": {},
            "missing_workflows": list(REQUIRED_WORKFLOWS),
            "unsuccessful_workflows": [],
        }

    runs = json.loads(stdout)
    workflow_runs: dict[str, dict[str, Any]] = {}
    for run in runs:
        name = run.get("name")
        if run.get("headSha") == sha and name in REQUIRED_WORKFLOWS and name not in workflow_runs:
            workflow_runs[name] = run

    missing = [name for name in REQUIRED_WORKFLOWS if name not in workflow_runs]
    unsuccessful = [
        name
        for name, run in workflow_runs.items()
        if not (run.get("status") == "completed" and run.get("conclusion") == "success")
    ]
    return {
        "available": True,
        "current_sha": sha,
        "workflow_runs": workflow_runs,
        "missing_workflows": missing,
        "unsuccessful_workflows": unsuccessful,
        "all_required_workflows_successful": not missing and not unsuccessful,
    }


def build_report() -> dict[str, Any]:
    """Build the deployment readiness report."""
    sha = current_sha()
    local = local_tool_status()
    github = github_workflow_status(sha)
    local_ready = local["all_local_deployment_tools_ready"]
    github_ready = github.get("all_required_workflows_successful", False)

    if local_ready:
        status = "PASS"
        verification_source = "local_tools_ready"
        reason = "Docker, Kind, kubectl, and Docker daemon are available locally."
    elif github_ready:
        status = "PASS"
        verification_source = "github_actions_current_sha"
        reason = (
            "Local deployment tools are not fully available, but Docker Build and "
            "Deploy Kind succeeded in GitHub Actions for the current SHA."
        )
    else:
        status = "BLOCKED_BY_LOCAL_SETUP"
        verification_source = "insufficient_deployment_evidence"
        reason = (
            "Local Docker/Kind/kubectl are not fully available and current-SHA "
            "GitHub deployment workflow evidence is missing or unsuccessful."
        )

    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "verification_source": verification_source,
        "reason": reason,
        "current_sha": sha,
        "local": local,
        "github_actions": github,
    }


def main() -> None:
    """Print deployment readiness and return non-zero only when evidence is insufficient."""
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
