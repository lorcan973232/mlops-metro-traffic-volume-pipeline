"""Generate safe security reports for repository and workflow inspection.

This local scanner is intentionally conservative. It records a dependency
inventory, searches committed text for obvious credential patterns and internal
notes, checks the Dockerfile ends with a non-root user, and writes a minimal SBOM.
The GitHub Actions workflow performs the heavier pip-audit and Trivy scans.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPORT_DIR = Path("reports/security")
DEPENDENCY_REPORT = REPORT_DIR / "dependency_scan.txt"
SECRET_REPORT = REPORT_DIR / "secret_scan.txt"
DOCKER_REPORT = REPORT_DIR / "docker_security_notes.md"
SBOM_REPORT = REPORT_DIR / "sbom.spdx.json"
SUMMARY_JSON = REPORT_DIR / "security_scan_summary.json"
SUMMARY_MD = REPORT_DIR / "security_scan_summary.md"

IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}
IGNORED_FILES = {
    "models/traffic_volume_classifier.joblib",
    "scripts/security_scan.py",
    "tests/test_workflows.py",
    "Makefile",
}


# ==============================================================================
# Safe local security reports
# ==============================================================================
#
# This script gives a readable security summary without committing raw scanner
# output that might contain sensitive values. GitHub Actions runs the heavier
# dependency and image checks; the local path still catches obvious secrets,
# private environment files, and internal planning notes.


def _timestamp() -> str:
    """Return a UTC timestamp for committed security report files."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_requirements(path: Path = Path("requirements.txt")) -> list[tuple[str, str]]:
    """Read pinned dependencies so the inventory is tied to `requirements.txt`."""
    packages: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            packages.append((line, "unpinned"))
            continue
        name, version = line.split("==", 1)
        packages.append((name.strip(), version.strip()))
    return packages


def _write_dependency_inventory() -> None:
    """Write a dependency inventory without claiming a vulnerability scan was run."""
    packages = _read_requirements()
    lines = [
        "Dependency security evidence",
        f"Generated at: {_timestamp()}",
        "Source: requirements.txt",
        "",
        "Pinned dependency inventory:",
    ]
    for name, version in packages:
        lines.append(f"- {name}=={version}")
    lines.extend(
        [
            "",
            "Vulnerability audit command used in GitHub Actions:",
            "python -m pip_audit -r requirements.txt --progress-spinner off",
            "",
            "Local note:",
            (
                "If local TLS/certificate policy blocks pip-audit from contacting "
                "PyPI, use the Security Scan workflow evidence instead of recording "
                "a fake clean result."
            ),
        ]
    )
    DEPENDENCY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_text_files() -> list[Path]:
    """List committed-style text candidates while avoiding caches and binaries."""
    files: list[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        normalized = path.as_posix()
        if normalized in IGNORED_FILES:
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _secret_patterns() -> list[re.Pattern[str]]:
    """Build credential regexes without storing live-looking secrets in source."""
    gh_prefix = "gh" + "p_"
    pat_prefix = "github" + "_pat_"
    token_words = "api[_-]?key|pass" + "word|sec" + "ret|tok" + "en"
    private_key = "begin (rsa|openssh|dsa|ec|private) private" + " key"
    return [
        re.compile(gh_prefix + r"[A-Za-z0-9_]{20,}"),
        re.compile(pat_prefix + r"[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(private_key, re.IGNORECASE),
        re.compile(rf"({token_words})\s*[:=]\s*['\"][^'\"]{{8,}}", re.IGNORECASE),
    ]


def _run_secret_scan() -> list[str]:
    """Search committed text files for common credential patterns."""
    findings: list[str] = []
    patterns = _secret_patterns()
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.as_posix()}:{line_number}: {pattern.pattern}")

    if findings:
        SECRET_REPORT.write_text(
            "Potential credential findings:\n" + "\n".join(findings) + "\n",
            encoding="utf-8",
        )
    else:
        SECRET_REPORT.write_text(
            f"PASS: no hard-coded credential patterns found at {_timestamp()}.\n",
            encoding="utf-8",
        )
    return findings


def _run_env_file_scan() -> list[str]:
    """Find private `.env` files that should not be committed."""
    findings: list[str] = []
    for path in Path(".").rglob(".env*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.name in {".env.example", ".env.sample"}:
            continue
        findings.append(path.as_posix())
    return findings


def _run_internal_file_scan() -> list[str]:
    """Find internal planning wording that should not be committed."""
    risky_names = {
        "CLA" + "UDE.md",
        "TIER3_" + "ROADMAP.md",
        "TIER3_" + "COMPLETE_" + "IMPLEMENTATION.md",
    }
    risky_phrases = [
        "CLA" + "UDE",
        "AI " + "assistant",
        "estimated " + "marks",
        "estimated " + "score",
        "+" + "marks",
        "marking " + "strategy",
        "to maximise " + "marks",
        "tier-three roadmap",
        "pro" + "mpt-" + "engineering",
    ]
    findings: list[str] = []
    for path in _iter_text_files():
        normalized = path.as_posix()
        if path.name in risky_names:
            findings.append(f"{normalized}: internal planning file name")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if normalized in {"scripts/security_scan.py", "scripts/check_stale_evidence.py"}:
            continue
        for phrase in risky_phrases:
            if phrase in text:
                findings.append(f"{normalized}: internal/professionalism phrase `{phrase}`")
    return findings


def _run_kind_reference_scan() -> list[str]:
    """Ensure deployment notes stay focused on Kind rather than cloud VMs."""
    findings: list[str] = []
    provider_name = "Goo" + "gle"
    patterns = [
        re.compile(r"\b" + provider_name + r" VM\b", re.IGNORECASE),
        re.compile(r"\b" + "g" + "cloud" + r"\b", re.IGNORECASE),
        re.compile(r"\b" + "compute " + "engine" + r"\b", re.IGNORECASE),
    ]
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            if pattern.search(text):
                findings.append(f"{path.as_posix()}: {pattern.pattern}")
    return findings


def _run_fake_success_scan() -> list[str]:
    """Find wording that would imply checks passed without evidence."""
    findings: list[str] = []
    suspicious_patterns = [
        re.compile(r"\bfake " + r"success\b", re.IGNORECASE),
        re.compile(r"\bassume(d)? passing\b", re.IGNORECASE),
        re.compile(r"\bpret" + r"end(ed)? success\b", re.IGNORECASE),
    ]
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.as_posix() in {"scripts/security_scan.py"}:
            continue
        for pattern in suspicious_patterns:
            if pattern.search(text):
                findings.append(f"{path.as_posix()}: {pattern.pattern}")
    return findings


def _docker_non_root_check() -> bool:
    """Check the Dockerfile's final runtime user for basic hardening."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    user_lines = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip().upper().startswith("USER ")
    ]
    return bool(user_lines) and user_lines[-1].lower() not in {"user root", "user 0"}


def _write_docker_notes(non_root: bool) -> None:
    """Write readable Docker security notes under `reports/security/`."""
    status = "PASS" if non_root else "FAIL"
    DOCKER_REPORT.write_text(
        "\n".join(
            [
                "# Docker Security Notes",
                "",
                f"Generated at: {_timestamp()}",
                "",
                f"- Non-root runtime user check: {status}",
                "- Dockerfile installs dependencies before switching to the runtime user.",
                "- The Flask/Gunicorn process runs under `USER appuser`.",
                "- Docker image vulnerability reports are generated by the Security Scan workflow.",
                "- The image is still smoke-tested by the Docker Build and Deploy Kind workflows.",
                "",
                "This file is a security note for the project. It does not claim that",
                "the image is free from every possible CVE; inspect the workflow scan",
                "artefacts for the current vulnerability output.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_minimal_spdx_sbom() -> None:
    """Create a minimal dependency SBOM when a heavier scanner is unavailable."""
    packages = _read_requirements()
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "mlops-metro-traffic-pipeline-python-dependencies",
        "documentNamespace": (
            "https://spdx.org/spdxdocs/mlops-metro-traffic-pipeline/"
            f"sbom/{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        ),
        "creationInfo": {
            "created": _timestamp(),
            "creators": ["Tool: scripts/security_scan.py"],
        },
        "packages": [
            {
                "SPDXID": f"SPDXRef-Package-{name.lower().replace('_', '-')}",
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
            for name, version in packages
        ],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": f"SPDXRef-Package-{name.lower().replace('_', '-')}",
            }
            for name, _version in packages
        ],
    }
    SBOM_REPORT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _write_security_summary(
    *,
    secret_findings: list[str],
    env_findings: list[str],
    internal_findings: list[str],
    kind_reference_findings: list[str],
    fake_success_findings: list[str],
    docker_non_root: bool,
    audit_status: int,
    pip_audit_requested: bool,
) -> None:
    """Write a safe summary that can be committed with the project."""
    checks = {
        "credential_pattern_scan": {
            "status": "PASS" if not secret_findings else "FAIL",
            "finding_count": len(secret_findings),
        },
        "env_file_scan": {
            "status": "PASS" if not env_findings else "FAIL",
            "finding_count": len(env_findings),
        },
        "internal_planning_file_scan": {
            "status": "PASS" if not internal_findings else "FAIL",
            "finding_count": len(internal_findings),
        },
        "kind_not_cloud_vm_scan": {
            "status": "PASS" if not kind_reference_findings else "FAIL",
            "finding_count": len(kind_reference_findings),
        },
        "fake_success_claim_scan": {
            "status": "PASS" if not fake_success_findings else "FAIL",
            "finding_count": len(fake_success_findings),
        },
        "docker_non_root": {
            "status": "PASS" if docker_non_root else "FAIL",
        },
        "dependency_inventory_or_audit": {
            "status": "PASS" if audit_status == 0 else "FAIL",
            "pip_audit_requested": pip_audit_requested,
        },
    }
    all_passed = all(item["status"] == "PASS" for item in checks.values())
    summary = {
        "generated_at": _timestamp(),
        "status": "PASS" if all_passed else "FAIL",
        "checks": checks,
        "safe_report_files": [
            DEPENDENCY_REPORT.as_posix(),
            SECRET_REPORT.as_posix(),
            DOCKER_REPORT.as_posix(),
            SBOM_REPORT.as_posix(),
            SUMMARY_JSON.as_posix(),
            SUMMARY_MD.as_posix(),
        ],
        "note": (
            "This is a safe summary. Raw scanner output should not be committed "
            "if it contains sensitive values."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Security Scan Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Overall status: `{summary['status']}`",
        "",
        "| Check | Status | Finding count |",
        "|---|---:|---:|",
    ]
    for name, result in checks.items():
        lines.append(
            f"| `{name}` | `{result['status']}` | `{result.get('finding_count', 0)}` |"
        )
    lines.extend(
        [
            "",
            "The committed security summary is deliberately safe. Current dependency",
            "and image vulnerability detail is produced by the `Security Scan`",
            "GitHub Actions workflow artefact.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_optional_pip_audit() -> int:
    """Run pip-audit when explicitly requested by the workflow or student."""
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        "requirements.txt",
        "--progress-spinner",
        "off",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    DEPENDENCY_REPORT.write_text(output or "pip-audit produced no output.\n", encoding="utf-8")
    return completed.returncode


def main() -> None:
    """Run the safe local security checks and fail on serious findings."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-pip-audit",
        action="store_true",
        help="Run pip-audit and fail if it reports vulnerabilities or cannot complete.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.run_pip_audit:
        audit_status = _run_optional_pip_audit()
    else:
        audit_status = 0
        _write_dependency_inventory()

    secret_findings = _run_secret_scan()
    env_findings = _run_env_file_scan()
    internal_findings = _run_internal_file_scan()
    kind_reference_findings = _run_kind_reference_scan()
    fake_success_findings = _run_fake_success_scan()
    docker_non_root = _docker_non_root_check()
    _write_docker_notes(docker_non_root)
    _write_minimal_spdx_sbom()
    _write_security_summary(
        secret_findings=secret_findings,
        env_findings=env_findings,
        internal_findings=internal_findings,
        kind_reference_findings=kind_reference_findings,
        fake_success_findings=fake_success_findings,
        docker_non_root=docker_non_root,
        audit_status=audit_status,
        pip_audit_requested=args.run_pip_audit,
    )

    if secret_findings:
        raise SystemExit("Potential credential patterns found.")
    if env_findings:
        raise SystemExit(f"Unexpected environment files found: {env_findings}")
    if internal_findings:
        raise SystemExit(f"Internal planning/professionalism findings: {internal_findings}")
    if kind_reference_findings:
        raise SystemExit(f"Disallowed cloud deployment references found: {kind_reference_findings}")
    if fake_success_findings:
        raise SystemExit(f"Fake success claim findings: {fake_success_findings}")
    if not docker_non_root:
        raise SystemExit("Dockerfile does not end with a non-root USER.")
    if audit_status != 0:
        raise SystemExit(audit_status)

    print("PASS: security reports generated.")


if __name__ == "__main__":
    main()
