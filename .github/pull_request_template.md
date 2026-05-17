## Artefact Evidence

<!-- This checklist asks the author to connect a PR to executable evidence.
     It is included because MLOps changes should be proved by tests, reports, or
     workflows, not only described in prose. -->

- [ ] Dataset ingestion, preprocessing, training, API, Docker, Kind, CI/CD, CT, or CM code changed as needed.
- [ ] Tests added or updated for the changed behaviour.
- [ ] `pytest` passed locally or in CI.
- [ ] No pipeline stage is described only in documentation without executable evidence.
