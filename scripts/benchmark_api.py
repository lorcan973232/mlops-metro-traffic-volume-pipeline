"""Measure Flask prediction latency and write an SLA evidence report.

The Docker workflow and local demos use this script after the API is already
running. It sends the same valid example payload repeatedly, records latency
percentiles, and writes `reports/benchmarks/api_sla_report.json` so performance
evidence is based on real requests rather than a static claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import PredictionRequestExample


def benchmark_api(api_url: str, samples: int = 100, warmup_samples: int = 10) -> dict:
    """Benchmark `/predict` latency and compute marker-readable percentiles."""
    example_payload = json.dumps(PredictionRequestExample().as_payload()).encode("utf-8")
    endpoint = f"{api_url}/predict"
    sla_threshold_ms = 200
    latencies: list[float] = []

    print(f"Warming up with {warmup_samples} requests...")
    for i in range(warmup_samples):
        try:
            start = time.perf_counter()
            req = Request(
                endpoint,
                data=example_payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as response:
                response.read()
            latency_ms = (time.perf_counter() - start) * 1000
            if i % max(1, warmup_samples // 3) == 0:
                print(f"  Warmup {i+1}/{warmup_samples}: {latency_ms:.1f}ms")
        except URLError as exc:
            print(f"Error during warmup: {exc}", file=sys.stderr)
            raise

    print(f"\nBenchmarking {samples} requests...")
    for i in range(samples):
        try:
            start = time.perf_counter()
            req = Request(
                endpoint,
                data=example_payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as response:
                response.read()
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            if (i + 1) % max(1, samples // 5) == 0:
                print(f"  Progress {i+1}/{samples}, avg: {statistics.mean(latencies[-10:]):.1f}ms")
        except URLError as exc:
            print(f"Error during benchmark: {exc}", file=sys.stderr)
            raise

    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    mean_latency = statistics.mean(latencies)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0

    report = {
        "status": "benchmarked",
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "endpoint": endpoint,
        "api_url": api_url,
        "sla_threshold_ms": sla_threshold_ms,
        "samples": samples,
        "warmup_samples": warmup_samples,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean_latency, 2),
        "stdev_ms": round(std_dev, 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "sla_met": p99 < sla_threshold_ms,
    }

    return report


def main() -> None:
    """Parse CLI options, run the benchmark, and fail if the p99 SLA is missed."""
    parser = argparse.ArgumentParser(description="Benchmark Flask API latency and SLA compliance")
    parser.add_argument("api_url", help="Base URL of the API (e.g., http://127.0.0.1:5000)")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to benchmark (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup samples (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/benchmarks/api_sla_report.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    print(f"Benchmarking API at {args.api_url}")
    print(f"Samples: {args.samples}, Warmup: {args.warmup}\n")

    report = benchmark_api(args.api_url, samples=args.samples, warmup_samples=args.warmup)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBenchmark Results:")
    print(f"  P50 (50th percentile): {report['p50_ms']}ms")
    print(f"  P95 (95th percentile): {report['p95_ms']}ms")
    print(f"  P99 (99th percentile): {report['p99_ms']}ms")
    print(f"  Mean: {report['mean_ms']}ms")
    print(f"  Stdev: {report['stdev_ms']}ms")
    print(
        f"  SLA (P99 < {report['sla_threshold_ms']}ms): "
        f"{'PASS' if report['sla_met'] else 'FAIL'}"
    )
    print(f"\nReport saved to: {output_path}")

    if not report["sla_met"]:
        print(
            f"WARNING: SLA not met! P99 is {report['p99_ms']}ms, "
            f"threshold is {report['sla_threshold_ms']}ms"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
