# Branching Evidence

This file records real branch and pull-request evidence for the artefact branching strategy.

## Strategy Mapping

| Branch | Evidence role |
|---|---|
| `main` | Stable release branch. Final artefact commits are promoted here and trigger CI, Docker Build, Kind deployment, visibility checks, and Security Scan. |
| `develop` | Integration branch. Feature work is merged here before promotion to `main`. |
| `feature/final-artefact-verification` | Earlier evidence branch proving the documented `feature/* -> develop` pull-request path. |
| `feature/final-artefact-hardening` | Final artefact-hardening branch proving the same path after security, Docker, reproducibility, CT/CM, and evidence updates. |

## Current Branch Evidence

| Evidence item | Value |
|---|---|
| Repository | <https://github.com/lorcan973232/mlops-wine-quality-pipeline> |
| Default branch | `main` |
| Current final feature branch | `feature/final-artefact-hardening` |
| Current feature SHA | `c77d4cf38cea994a249530f95d3fd97f39487cf1` |
| Current `main` SHA before final promotion | `681e1388d834a0ffd130e51ca1f4acc3b09fc1dc` |
| Current `develop` SHA after PR #3 merge | `4b5f42d6a3620db189404c5fbda5a8f7fe3f0444` |
| Develop synced with main before PR #4 merge | No. PR #4 promotes `develop` to `main`; after merge, `develop` is fast-forwarded to the final `main` commit and checked with `git rev-list --left-right --count origin/develop...origin/main`. |

## Pull Request Evidence

| Evidence item | Value |
|---|---|
| Earlier feature-to-develop PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/1> |
| Earlier feature PR status | Merged into `develop` at `2026-05-16T12:52:43Z`; CI and Docker checks passed. |
| Earlier develop-to-main PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/2> |
| Earlier develop-to-main status | Merged into `main` at `2026-05-16T12:55:42Z`; CI and Docker checks passed. |
| Final hardening feature-to-develop PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/3> |
| Final hardening PR status | Merged into `develop` at `2026-05-16T14:30:31Z`; PR checks passed for CI, Data Preprocessing, Train and Evaluate, Docker Build, and Security Scan after vulnerable dependency pins were fixed. |
| Final hardening PR merge commit | `4b5f42d6a3620db189404c5fbda5a8f7fe3f0444` |
| Final develop-to-main PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/4> |
| Final develop-to-main PR status | Open at the time this evidence was written; this PR completes the `feature/final-artefact-hardening -> develop -> main` promotion path. |
| Strategy mapping | The artefact exercises `feature/* -> develop -> main`, matching the README branching strategy. |

## CI Status Recorded For PR #3

PR #3 passed these GitHub checks after the dependency-vulnerability fix commit:

- CI: `Python setup check` and `Code, test, and ML quality gates`.
- Data Preprocessing: `Ingest and validate public dataset` and `Preprocess deterministic classification dataset`.
- Train and Evaluate: `Prepare selected public dataset` and `Train, evaluate, and register candidate model`.
- Docker Build: `Build image tagged with commit SHA`.
- Security Scan: `Secrets, dependencies, Docker, and SBOM evidence`.

This file must be read with the latest GitHub PR and Actions pages for the final SHA. It records evidence and does not claim future public visibility or future workflow success before those checks exist.
