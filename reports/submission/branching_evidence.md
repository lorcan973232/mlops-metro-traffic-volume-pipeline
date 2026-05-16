# Branching Evidence

This file records real branch and pull-request evidence for the artefact branching strategy.

## Strategy Mapping

| Branch | Evidence role |
|---|---|
| `main` | Stable release branch. Final artefact commits are pushed here and trigger CI, Docker Build, Train and Evaluate, Deploy Kind, and visibility checks. |
| `develop` | Integration branch. Feature work is merged here before promotion to `main`. |
| `feature/final-artefact-verification` | Evidence branch used to prove the documented `feature/* -> develop` pull-request path. |

## Current Evidence

This file is intentionally committed before the final branch exercise. The final post-push verifier updates this section with:

- branch SHAs;
- feature-to-develop PR URL;
- develop-to-main PR URL or documented promotion path;
- CI status;
- merge status.

No success is claimed here until the GitHub pull requests and checks actually exist.
