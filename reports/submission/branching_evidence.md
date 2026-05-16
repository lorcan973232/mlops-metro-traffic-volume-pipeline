# Branching Evidence

This file records real branch and pull-request evidence for the artefact branching strategy.

## Strategy Mapping

| Branch | Evidence role |
|---|---|
| `main` | Stable release branch. Final artefact commits are pushed here and trigger CI, Docker Build, Train and Evaluate, Deploy Kind, and visibility checks. |
| `develop` | Integration branch. Feature work is merged here before promotion to `main`. |
| `feature/final-artefact-verification` | Evidence branch used to prove the documented `feature/* -> develop` pull-request path. |

## Current Evidence

Feature branch evidence was created from `develop` for the final artefact verification exercise.

| Evidence item | Value |
|---|---|
| `main` branch | Stable release branch; develop-to-main PR #2 was merged into `main` |
| `develop` branch | Created from `main`, pushed to `origin/develop`, and used as the integration branch |
| Feature branch | `feature/final-artefact-verification`, pushed to `origin/feature/final-artefact-verification` |
| Feature branch purpose | Safe evidence-only change proving the `feature/* -> develop` path |
| Feature-to-develop PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/1> |
| Feature PR CI status | Passed: `Python setup check`, `Code, test, and ML quality gates`, and `Build image tagged with commit SHA` completed successfully |
| Feature PR merge status | Merged into `develop` at `2026-05-16T12:52:43Z`; merge commit `3ec34d1ba555b958a1104af5626f77ac1e38dcc3` |
| Develop-to-main PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/2> |
| Develop-to-main CI status | Passed: `Python setup check`, `Code, test, and ML quality gates`, and `Build image tagged with commit SHA` completed successfully |
| Develop-to-main merge status | Merged into `main` at `2026-05-16T12:55:42Z`; merge commit `4c8e5907567eca64a9b254dcc590fe12449f8b8a` |
| Strategy mapping | The artefact exercised `feature/final-artefact-verification -> develop -> main`, matching the README branching strategy |

All PR success, CI success, and merge success claims in this file are backed by the linked GitHub pull requests and check runs.
