"""Build the current final-readiness evidence pack for submission review.

This script is run manually or by the Final Readiness workflow near submission.
It gathers repository visibility, current SHA, recent GitHub Actions runs, core
report presence, Docker/Kind pointers, API/UI evidence, security evidence, and
Windows/Bash route status. Generated files are not committed because they become
stale after the next push.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from check_repo_visibility import build_evidence

REPORT_DIR = Path("reports/final_readiness")
GENERATED_DIR = REPORT_DIR / "generated"
REPORT_PATH = GENERATED_DIR / "final_readiness_report.json"
ACTIONS_PATH = GENERATED_DIR / "latest_github_actions_runs.json"
COMMANDS_PATH = GENERATED_DIR / "local_command_results.json"
SUMMARY_PATH = REPORT_DIR / "final_readiness_summary.md"
DEMO_PATH = REPORT_DIR / "live_demo_checklist.md"

EXPECTED_WORKFLOWS = {
    "CI",
    "Data Preprocessing",
    "Train and Evaluate",
    "Docker Build",
    "Continuous Training",
    "Deploy Kind",
    "Monitoring",
    "Tier 3 Model Analysis",
    "Security Scan",
    "Repository Visibility Check",
    "Bash Script Verification",
    "Final Readiness",
}


# ==============================================================================
# Final readiness evidence
# ==============================================================================
#
# The committed README explains the artefact, but this script creates current
# evidence for the exact checkout being assessed. It records the SHA, repository
# visibility, expected workflow runs, and evidence-file presence under
# `reports/final_readiness/generated/` so the marker is not relying on stale text.


def run_command(args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run a local command without crashing the readiness report on blockers."""
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON evidence file and report missing or invalid files clearly."""
    file_path = Path(path)
    if not file_path.is_file():
        return {"status": "missing", "path": str(file_path)}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid_json", "path": str(file_path), "error": str(exc)}


def latest_sha() -> str:
    """Return the current commit SHA used to tie evidence to this checkout."""
    code, stdout, _ = run_command(["git", "rev-parse", "HEAD"])
    return stdout if code == 0 and stdout else "unknown"


def repository_slug() -> str:
    """Reuse the visibility evidence builder to identify the GitHub repository."""
    visibility = build_evidence()
    return visibility.get("full_name", "")


def github_runs(repo: str) -> list[dict[str, Any]]:
    """Fetch recent GitHub Actions runs when GitHub CLI authentication is available."""
    if not repo:
        return []
    code, stdout, _ = run_command(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--branch",
            "main",
            "--limit",
            "50",
            "--json",
            "databaseId,workflowName,displayTitle,headSha,status,conclusion,createdAt,updatedAt,event,url",
        ],
        timeout=120,
    )
    if code != 0 or not stdout:
        return []
    return json.loads(stdout)


def workflow_summary(runs: list[dict[str, Any]], sha: str) -> dict[str, Any]:
    """Summarise whether each expected workflow has a run for the current SHA."""
    by_workflow: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.get("headSha") != sha:
            continue
        name = run.get("workflowName", "")
        if name not in by_workflow:
            by_workflow[name] = run
    missing = sorted(EXPECTED_WORKFLOWS - set(by_workflow))
    unsuccessful = sorted(
        name
        for name, run in by_workflow.items()
        if name in EXPECTED_WORKFLOWS
        and not (run.get("status") == "completed" and run.get("conclusion") == "success")
    )
    return {
        "latest_sha": sha,
        "expected_workflows": sorted(EXPECTED_WORKFLOWS),
        "latest_sha_runs": by_workflow,
        "missing_expected_workflows_current_sha": missing,
        "unsuccessful_expected_workflows_current_sha": unsuccessful,
        "all_expected_workflows_successful_current_sha": not missing and not unsuccessful,
    }


def bash_status() -> dict[str, Any]:
    """Record whether Windows Bash is usable or correctly marked as blocked."""
    code, stdout, stderr = run_command(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts/check_bash_environment.ps1"],
        timeout=60,
    )
    marker = "LOCAL_BASH_BLOCKED_BY_WINDOWS_WSL_SETUP"
    return {
        "command": "powershell -ExecutionPolicy Bypass -File scripts/check_bash_environment.ps1",
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "local_bash_blocked_by_windows_wsl_setup": marker in stdout or marker in stderr,
        "handled": code == 0 or marker in stdout or marker in stderr,
    }


def command_results() -> dict[str, Any]:
    """Summarise local evidence checks without rerunning every heavy command."""
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "note": (
            "This file summarises evidence generated by the dedicated verification "
            "commands and reports. Heavy local commands are run separately in final audit."
        ),
        "python_pipeline_reports_present": {
            "data_ingestion": Path("reports/metrics/data_ingestion.json").is_file(),
            "preprocessing": Path("reports/metrics/preprocessing.json").is_file(),
            "latest_metrics": Path("reports/metrics/latest_metrics.json").is_file(),
            "model_metadata": Path("reports/metrics/model_metadata.json").is_file(),
            "quality_gate": Path("reports/metrics/quality_gate_report.json").is_file(),
            "monitoring": Path("reports/monitoring/monitoring_report.json").is_file(),
            "drift": Path("reports/monitoring/drift_report.json").is_file(),
            "security": Path("reports/security/secret_scan.txt").is_file(),
        },
        "bash_environment": bash_status(),
        "powershell_path": {
            "check_setup_script": "scripts/check_setup.ps1",
            "deployment_scripts": [
                "scripts/create_kind_cluster.ps1",
                "scripts/deploy_kind.ps1",
                "scripts/smoke_test_api.ps1",
            ],
            "status": "supported_and_verified_by_final_audit_when_run",
        },
    }


def build_report() -> dict[str, Any]:
    """Build the SHA-specific readiness report used before submission and demo."""
    visibility = build_evidence()
    sha = latest_sha()
    repo = visibility.get("full_name", repository_slug())
    runs = github_runs(repo)
    actions = workflow_summary(runs, sha)
    metrics = read_json("reports/metrics/latest_metrics.json")
    gate = read_json("reports/metrics/quality_gate_report.json")
    monitoring = read_json("reports/monitoring/monitoring_report.json")
    drift = read_json("reports/monitoring/drift_report.json")
    metadata = read_json("reports/metrics/model_metadata.json")

    report = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "repository": visibility,
        "latest_commit_sha": sha,
        "github_actions": actions,
        "ml_pipeline": {
            "latest_metrics_status": metrics.get("status"),
            "metric_summary": metrics.get("metric_summary", metrics),
            "quality_gate_passed": gate.get("passed"),
            "quality_gate_decision": gate.get("decision"),
            "model_version": metadata.get("model_version"),
        },
        "docker": {
            "dockerfile_present": Path("Dockerfile").is_file(),
            "docker_workflow": actions["latest_sha_runs"].get("Docker Build", {}),
        },
        "kind": {
            "manifests_present": Path("deployment/kind/deployment.yaml").is_file()
            and Path("deployment/kind/service.yaml").is_file(),
            "deploy_workflow": actions["latest_sha_runs"].get("Deploy Kind", {}),
        },
        "flask_api": {
            "app_module_present": Path("app/main.py").is_file(),
            "smoke_scripts": {
                "powershell": Path("scripts/smoke_test_api.ps1").is_file(),
                "bash": Path("scripts/smoke_test_api.sh").is_file(),
            },
        },
        "ui": {
            "index_template_present": Path("app/templates/index.html").is_file(),
            "static_js_present": Path("app/static/app.js").is_file(),
            "static_css_present": Path("app/static/style.css").is_file(),
        },
        "monitoring": {
            "monitoring_status": monitoring.get("status"),
            "monitoring_mode": monitoring.get("monitoring_mode"),
            "drift_status": drift.get("status"),
            "drift_metric": drift.get("drift_metric"),
            "retraining_required": drift.get("retraining_required"),
        },
        "security": {
            "secret_scan_present": Path("reports/security/secret_scan.txt").is_file(),
            "dependency_scan_present": Path("reports/security/dependency_scan.txt").is_file(),
            "sbom_present": Path("reports/security/sbom.spdx.json").is_file(),
            "security_workflow": actions["latest_sha_runs"].get("Security Scan", {}),
        },
        "bash_and_windows_support": command_results()["bash_environment"],
        "public_until_21_june_2026_note": visibility["future_compliance_note"],
        "remaining_student_responsibilities": [
            "Keep the repository public until 21 June 2026.",
            "Rerun the visibility workflow close to submission and before the live demo.",
            "Use PowerShell on Windows if local bash resolves to a broken WSL installation.",
        ],
    }
    return report


def write_summary(report: dict[str, Any]) -> None:
    """Write a stable human summary that points to generated current evidence."""
    lines = [
        "# Final Artefact Readiness Summary",
        "",
        "This committed file is a stable description of the readiness evidence.",
        "It deliberately avoids hard-coded commit SHAs and workflow conclusions.",
        "",
        "Current SHA-specific proof is generated by `python scripts/final_readiness_check.py`",
        "under `reports/final_readiness/generated/` and by the `Final Readiness`",
        "GitHub Actions workflow artefact. The generated directory is local/workflow",
        "output and is not committed because it becomes stale after the next commit.",
        "",
        "The readiness check records:",
        "",
        "- repository URL and current public visibility evidence;",
        "- current commit SHA at the time of the check;",
        "- latest GitHub Actions run snapshot for the expected workflows;",
        "- ML metrics, CT quality gate, monitoring, drift, Docker, Kind, API, UI,",
        "  security, Bash, and PowerShell evidence pointers;",
        "- remaining student responsibilities for public visibility and live demo use.",
        "",
        "Before final submission and before the live demo, rerun these workflows:",
        "",
        "- `Final Readiness`;",
        "- `Repository Visibility Check`;",
        "- `Security Scan`;",
        "- `Docker Build`;",
        "- `Deploy Kind`;",
        "- `Continuous Training`;",
        "- `Monitoring`.",
        "",
        (
            "The future public-until date is not claimed as permanently proven. "
            "It is safeguarded by the visibility script, scheduled workflow, "
            "evidence report, and explicit student responsibility to keep the "
            "repository public until 21 June 2026."
        ),
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_demo_checklist() -> None:
    """Write a concise live-demo checklist without pretending the video is done."""
    DEMO_PATH.write_text(
        "\n".join(
            [
                "# Live Demo Checklist",
                "",
                "1. Show the public GitHub repository and latest commit SHA.",
                (
                    "2. Show the latest successful Actions runs for CI, data, "
                    "train/evaluate, Docker, Deploy Kind, CT, Monitoring, Tier 3 "
                    "analysis, Security, Repository Visibility, and Bash Script "
                    "Verification."
                ),
                (
                    "3. Open `reports/submission/public_repository_evidence.json` "
                    "and state that the repository must remain public until "
                    "21 June 2026."
                ),
                (
                    "4. Run `powershell -ExecutionPolicy Bypass -File "
                    "scripts/check_setup.ps1` on Windows."
                ),
                (
                    "5. If Windows bash is broken, run `powershell -ExecutionPolicy "
                    "Bypass -File scripts/check_bash_environment.ps1` and show the "
                    "documented fallback."
                ),
                (
                    "6. Run the Python pipeline, tests, lint, Docker build, Kind "
                    "deployment, smoke tests, monitoring, and security checks from "
                    "the README."
                ),
                (
                    "7. Open the Flask UI locally, use the example payload, and show "
                    "the `/health` and `/predict` smoke-test responses."
                ),
                "8. Show the CT quality gate and monitoring/drift reports.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate all final-readiness evidence files for this checkout."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    runs = github_runs(report["repository"].get("full_name", ""))
    commands = command_results()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ACTIONS_PATH.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    COMMANDS_PATH.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    write_summary(report)
    write_demo_checklist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
