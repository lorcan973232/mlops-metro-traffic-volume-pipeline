# Kind Deployment

This artefact uses Kind Kubernetes only for deployment evidence. It deploys the Flask
prediction image `mlops-flask-api:latest`, which serves the UCI Metro Interstate
Traffic Volume classifier at `models/traffic_volume_classifier.joblib`.

Kind is used because the marker can reproduce it on a local Docker machine or
inspect the GitHub Actions run without needing a persistent cloud account. The
deployment is deliberately ephemeral: the image is built from this repository,
loaded into the local Kind cluster, rolled out, port-forwarded, and smoke-tested.

Run from the repository root:

```bash
scripts/create_kind_cluster.sh
scripts/deploy_kind.sh
kubectl port-forward service/mlops-flask-api 8080:80
scripts/smoke_test_api.sh http://127.0.0.1:8080
```

PowerShell equivalents:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1
powershell -ExecutionPolicy Bypass -File scripts/deploy_kind.ps1
kubectl port-forward service/mlops-flask-api 8080:80
powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:8080
```

Open the live-demo UI at:

```text
http://127.0.0.1:8080/
```

The smoke test uses the real traffic feature schema: `temp`, `rain_1h`,
`snow_1h`, `clouds_all`, `hour`, `month`, `day_of_week`, `is_weekend`,
`is_holiday`, `weather_main`, `lag_1h_volume`, `lag_24h_volume`,
`lag_168h_volume`, `rolling_3h_volume`, and `rolling_24h_volume`.
