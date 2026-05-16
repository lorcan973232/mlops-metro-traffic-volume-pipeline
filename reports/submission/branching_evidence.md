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
| `main` branch | Created and pushed before this branch exercise |
| `develop` branch | Created from `main` and pushed to `origin/develop` |
| Feature branch | `feature/final-artefact-verification` |
| Feature branch purpose | Safe evidence-only change proving the `feature/* -> develop` path |
| Feature-to-develop PR | <https://github.com/lorcan973232/mlops-wine-quality-pipeline/pull/1> |
| Feature PR CI status | Passed: `Python setup check`, `Code, test, and ML quality gates`, and `Build image tagged with commit SHA` completed successfully |
| Feature PR merge status | Merged into `develop` at `2026-05-16T12:52:43Z`; merge commit `3ec34d1ba555b958a1104af5626f77ac1e38dcc3` |
| Develop-to-main PR | To be created from `develop` after this evidence update is pushed |
| Develop-to-main CI status | To be recorded from the real GitHub PR checks |
| Develop-to-main merge status | To be recorded only after the PR is actually merged |

No PR success, CI success, or merge success is claimed here until the GitHub pull requests and checks actually exist.
