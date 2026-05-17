from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

INTERNAL_FILENAMES = {
    "CLA" + "UDE.md",
    "TIER3_" + "ROADMAP.md",
    "TIER3_" + "COMPLETE_" + "IMPLEMENTATION.md",
}

INTERNAL_PHRASES = [
    "CLA" + "UDE",
    "AI " + "assistant",
    "estimated " + "marks",
    "estimated " + "score",
    "+" + "marks",
    "marking " + "strategy",
    "to maximise " + "marks",
    "Tier 3 " + "roadmap",
    "TIER3_" + "COMPLETE_" + "IMPLEMENTATION",
]

STALE_STATUS_PHRASES = [
    "Expected latest-" + "SHA workflows successful",
    "missing_expected_workflows_on_latest_" + "sha",
    "unsuccessful_expected_workflows_on_latest_" + "sha",
    "latest run snapshot from GitHub CLI",
    "Regenerate after final " + "push",
]

EXCLUDED_TEXT_SCAN_FILES = {
    "scripts/check_stale_evidence.py",
    "scripts/security_scan.py",
}


def run_command(args: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def tracked_files() -> list[Path]:
    code, stdout, stderr = run_command(["git", "ls-files"])
    if code != 0:
        raise SystemExit(f"git ls-files failed: {stderr}")
    return [Path(line) for line in stdout.splitlines() if line]


def check_internal_files(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for file_name in INTERNAL_FILENAMES:
        if Path(file_name).exists():
            findings.append(f"Internal planning file must not be submitted: {file_name}")
    return findings


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def check_internal_phrases(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in files:
        posix = path.as_posix()
        if posix in EXCLUDED_TEXT_SCAN_FILES:
            continue
        text = read_text(path)
        if text is None:
            continue
        for phrase in INTERNAL_PHRASES:
            if phrase in text:
                findings.append(f"{posix}: contains internal phrase `{phrase}`")
    return findings


def check_final_readiness_reports() -> list[str]:
    findings: list[str] = []
    sha_pattern = re.compile(r"\b[0-9a-f]{40}\b")
    report_dir = Path("reports/final_readiness")
    for path in report_dir.glob("*"):
        if not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            continue
        if sha_pattern.search(text):
            findings.append(f"{path.as_posix()}: committed readiness file contains a SHA")
        for phrase in STALE_STATUS_PHRASES:
            if phrase in text:
                findings.append(f"{path.as_posix()}: stale status phrase `{phrase}`")
    return findings


def check_public_visibility_snapshot() -> list[str]:
    path = Path("reports/submission/public_repository_evidence.json")
    if not path.is_file():
        return [f"{path.as_posix()}: missing public repository evidence"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path.as_posix()}: invalid JSON: {exc}"]

    findings: list[str] = []
    if payload.get("future_public_until_proven") is not False:
        findings.append(f"{path.as_posix()}: must not claim future public access is proven")
    scope = str(payload.get("evidence_scope", "")).lower()
    if "snapshot" not in scope:
        findings.append(f"{path.as_posix()}: must label visibility evidence as a snapshot")
    if payload.get("private") is not False or payload.get("visibility") != "public":
        findings.append(f"{path.as_posix()}: current visibility is not verified public")
    return findings


def main() -> None:
    files = tracked_files()
    findings = []
    findings.extend(check_internal_files(files))
    findings.extend(check_internal_phrases(files))
    findings.extend(check_final_readiness_reports())
    findings.extend(check_public_visibility_snapshot())

    if findings:
        print("FAIL: stale or internal evidence found.")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)

    print("PASS: no stale SHA claims, internal planning files, or misleading evidence found.")


if __name__ == "__main__":
    main()
