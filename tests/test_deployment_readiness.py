"""Tests for the deployment-readiness evidence helper."""

from __future__ import annotations

from pathlib import Path

from scripts import check_deployment_readiness


def test_deployment_readiness_script_reports_local_or_github_evidence() -> None:
    """Check the helper has the expected PASS/fallback contract."""
    report = check_deployment_readiness.build_report()

    assert report["status"] in {"PASS", "BLOCKED_BY_LOCAL_SETUP"}
    assert report["verification_source"] in {
        "local_tools_ready",
        "github_actions_current_sha",
        "insufficient_deployment_evidence",
    }
    assert "Docker Build" in check_deployment_readiness.REQUIRED_WORKFLOWS
    assert "Deploy Kind" in check_deployment_readiness.REQUIRED_WORKFLOWS
    assert "local" in report
    assert "github_actions" in report


def test_readme_documents_deployment_readiness_check() -> None:
    """Check the marker-facing deployment-readiness command is documented."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "python scripts/check_deployment_readiness.py" in readme
    assert "github_actions_current_sha" in readme
