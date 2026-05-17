# Live Demo Checklist

1. Show the public GitHub repository and latest commit SHA.
2. Show the latest successful Actions runs for CI, data, train/evaluate, Docker, Deploy Kind, CT, Monitoring, Tier 3 analysis, Security, Repository Visibility, and Bash Script Verification.
3. Open `reports/submission/public_repository_evidence.json` and state that the repository must remain public until 21 June 2026.
4. Run `powershell -ExecutionPolicy Bypass -File scripts/check_setup.ps1` on Windows.
5. If Windows bash is broken, run `powershell -ExecutionPolicy Bypass -File scripts/check_bash_environment.ps1` and show the documented fallback.
6. Run the Python pipeline, tests, lint, Docker build, Kind deployment, smoke tests, monitoring, and security checks from the README.
7. Open the Flask UI locally, use the example payload, and show the `/health` and `/predict` smoke-test responses.
8. Show the CT quality gate and monitoring/drift reports.
