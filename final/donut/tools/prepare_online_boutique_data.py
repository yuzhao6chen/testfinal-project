"""Convert Online Boutique experiment exports into Donut KPI CSV files.

The original Donut examples use one CSV per KPI series with columns:

    timestamp,value,label

This script converts the project export layout used in ``final/``:

* ``summary/*kpi.csv`` for JMeter business KPIs.
* ``raw_prometheus/*.csv`` for Prometheus platform KPIs.
* ``logs/fault_events.jsonl`` for fault windows used as labels.

Prometheus exports may contain duplicate samples at chunk boundaries.  For a
given series and second, duplicate values are averaged before writing output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BUSINESS_KPI_COLUMNS = (
    "request_count",
    "success_count",
    "error_count",
    "error_rate",
    "avg_elapsed_ms",
    "p95_elapsed_ms",
    "p99_elapsed_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Online Boutique KPI data for Donut."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the experiment export directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where Donut-format CSV files will be written.",
    )
    parser.add_argument(
        "--label-end",
        choices=("delete", "recovered"),
        default="recovered",
        help=(
            "End labels at fault deletion time or at the scheduled recovered "
            "time. Default: recovered."
        ),
    )
    parser.add_argument(
        "--prom-label-scope",
        choices=("service", "global"),
        default="service",
        help=(
            "Use service-specific labels for pod/deployment Prometheus series "
            "or mark every Prometheus series during every fault. Node-level "
            "series are always global. Default: service."
        ),
    )
    parser.add_argument(
        "--include-jmeter-endpoints",
        action="store_true",
        help=(
            "Also aggregate jmeter/trial_run.jtl by request label, producing "
            "per-endpoint business KPI series."
        ),
    )
    return parser.parse_args()


def parse_iso8601(value: str) -> float:
    value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value).timestamp()


def safe_name(value: str, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value[:max_len] or "series"


def load_fault_windows(root: Path, label_end: str) -> List[Dict[str, object]]:
    events_path = root / "logs" / "fault_events.jsonl"
    windows: List[Dict[str, object]] = []

    with events_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "fault_block_start":
                continue

            end_key = (
                "scheduled_delete"
                if label_end == "delete"
                else "scheduled_recovered"
            )
            windows.append(
                {
                    "phase_id": event["phase_id"],
                    "fault_type": event["fault_type"],
                    "target_service": event["target_service"],
                    "start": parse_iso8601(event["scheduled_apply"]),
                    "end": parse_iso8601(event[end_key]),
                }
            )

    if not windows:
        raise ValueError(f"No fault windows found in {events_path}")
    return windows


def is_labeled(
    timestamp: int,
    windows: Iterable[Dict[str, object]],
    series_service: Optional[str] = None,
) -> int:
    for window in windows:
        if series_service and window["target_service"] != series_service:
            continue
        if float(window["start"]) <= timestamp < float(window["end"]):
            return 1
    return 0


def write_kpi_csv(path: Path, rows: Iterable[Tuple[int, float, int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("timestamp", "value", "label"))
        for timestamp, value, label in rows:
            if math.isfinite(value):
                writer.writerow((timestamp, value, label))
                count += 1
    return count


def find_jmeter_kpi_file(root: Path) -> Path:
    candidates = sorted((root / "summary").glob("*kpi.csv"))
    if not candidates:
        raise FileNotFoundError("Cannot find summary/*kpi.csv")
    if len(candidates) > 1:
        names = ", ".join(str(p) for p in candidates)
        raise ValueError(f"Expected one JMeter KPI CSV, found: {names}")
    return candidates[0]


def convert_jmeter(
    root: Path,
    output: Path,
    windows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    source = find_jmeter_kpi_file(root)
    with source.open("r", newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))

    manifest_rows: List[Dict[str, object]] = []
    for column in BUSINESS_KPI_COLUMNS:
        rows = []
        for record in records:
            timestamp = int(float(record["bin_start"]))
            value = float(record[column])
            rows.append((timestamp, value, is_labeled(timestamp, windows)))

        out_path = output / "jmeter" / f"jmeter__{column}.csv"
        count = write_kpi_csv(out_path, rows)
        manifest_rows.append(
            {
                "file": str(out_path.relative_to(output)),
                "source": "jmeter",
                "metric": column,
                "series_service": "",
                "labels_json": "",
                "rows": count,
                "label_scope": "global",
            }
        )

    return manifest_rows


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def convert_jmeter_endpoints(
    root: Path,
    output: Path,
    windows: List[Dict[str, object]],
    step_seconds: int = 5,
) -> List[Dict[str, object]]:
    source = root / "jmeter" / "trial_run.jtl"
    if not source.exists():
        return []

    by_endpoint: Dict[str, Dict[int, Dict[str, object]]] = defaultdict(dict)
    with source.open("r", newline="", encoding="utf-8-sig") as f:
        for record in csv.DictReader(f):
            endpoint = record["label"]
            timestamp = int(int(record["timeStamp"]) / 1000)
            bin_start = (timestamp // step_seconds) * step_seconds
            bucket = by_endpoint[endpoint].setdefault(
                bin_start,
                {
                    "request_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "elapsed": [],
                    "latency": [],
                },
            )
            success = record["success"].strip().lower() == "true"
            bucket["request_count"] += 1
            bucket["success_count"] += int(success)
            bucket["error_count"] += int(not success)
            bucket["elapsed"].append(float(record["elapsed"]))
            bucket["latency"].append(float(record["Latency"]))

    metric_builders = {
        "request_count": lambda b: float(b["request_count"]),
        "success_count": lambda b: float(b["success_count"]),
        "error_count": lambda b: float(b["error_count"]),
        "error_rate": lambda b: (
            float(b["error_count"]) / float(b["request_count"])
            if b["request_count"]
            else 0.0
        ),
        "avg_elapsed_ms": lambda b: sum(b["elapsed"]) / len(b["elapsed"]),
        "p95_elapsed_ms": lambda b: percentile(b["elapsed"], 0.95),
        "p99_elapsed_ms": lambda b: percentile(b["elapsed"], 0.99),
        "avg_latency_ms": lambda b: sum(b["latency"]) / len(b["latency"]),
        "p95_latency_ms": lambda b: percentile(b["latency"], 0.95),
        "p99_latency_ms": lambda b: percentile(b["latency"], 0.99),
    }

    manifest_rows: List[Dict[str, object]] = []
    for endpoint, samples in sorted(by_endpoint.items()):
        endpoint_name = safe_name(endpoint)
        for metric_name, metric_builder in metric_builders.items():
            rows = []
            for timestamp, bucket in sorted(samples.items()):
                value = metric_builder(bucket)
                rows.append((timestamp, value, is_labeled(timestamp, windows)))

            out_path = (
                output
                / "jmeter_endpoint"
                / f"jmeter_endpoint__{endpoint_name}__{metric_name}.csv"
            )
            count = write_kpi_csv(out_path, rows)
            manifest_rows.append(
                {
                    "file": str(out_path.relative_to(output)),
                    "source": "jmeter_endpoint",
                    "metric": metric_name,
                    "series_service": "",
                    "labels_json": json.dumps(
                        {"request_label": endpoint}, ensure_ascii=False
                    ),
                    "rows": count,
                    "label_scope": "global",
                }
            )

    return manifest_rows


def parse_labels(labels_json: str) -> Dict[str, str]:
    try:
        parsed = json.loads(labels_json)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def service_from_labels(labels: Dict[str, str]) -> Optional[str]:
    if labels.get("deployment"):
        return labels["deployment"]

    pod = labels.get("pod")
    if pod:
        # Kubernetes pod names commonly follow deployment-replicaset-suffix.
        parts = pod.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:-2])
        return pod

    workload = labels.get("destination_workload")
    if workload:
        return workload

    return None


def series_file_name(metric_id: str, labels_json: str) -> str:
    labels = parse_labels(labels_json)
    service = service_from_labels(labels)
    if service:
        stem = f"{metric_id}__{service}"
    elif labels.get("instance"):
        stem = f"{metric_id}__{labels['instance']}"
    else:
        stem = f"{metric_id}__series"

    digest = hashlib.sha1(labels_json.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name(stem)}__{digest}.csv"


def convert_prometheus(
    root: Path,
    output: Path,
    windows: List[Dict[str, object]],
    label_scope: str,
) -> List[Dict[str, object]]:
    manifest_rows: List[Dict[str, object]] = []
    raw_dir = root / "raw_prometheus"

    for source in sorted(raw_dir.glob("*.csv")):
        by_series: Dict[str, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )
        metric_id = source.stem

        with source.open("r", newline="", encoding="utf-8-sig") as f:
            for record in csv.DictReader(f):
                labels_json = record["labels_json"]
                timestamp = int(float(record["timestamp"]))
                value = float(record["value"])
                if not math.isfinite(value):
                    continue
                bucket = by_series[labels_json][timestamp]
                bucket[0] += value
                bucket[1] += 1.0

        for labels_json, samples in sorted(by_series.items()):
            labels = parse_labels(labels_json)
            series_service = service_from_labels(labels)
            use_service_labels = label_scope == "service" and series_service
            label_service = series_service if use_service_labels else None

            rows = []
            for timestamp, (value_sum, value_count) in sorted(samples.items()):
                value = value_sum / value_count
                rows.append((timestamp, value, is_labeled(timestamp, windows, label_service)))

            out_path = output / "prometheus" / series_file_name(metric_id, labels_json)
            count = write_kpi_csv(out_path, rows)
            manifest_rows.append(
                {
                    "file": str(out_path.relative_to(output)),
                    "source": "prometheus",
                    "metric": metric_id,
                    "series_service": series_service or "",
                    "labels_json": labels_json,
                    "rows": count,
                    "label_scope": "service" if use_service_labels else "global",
                }
            )

    return manifest_rows


def write_manifest(output: Path, rows: List[Dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    fieldnames = (
        "file",
        "source",
        "metric",
        "series_service",
        "label_scope",
        "rows",
        "labels_json",
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    output = args.output.resolve()

    windows = load_fault_windows(root, args.label_end)
    manifest_rows = []
    manifest_rows.extend(convert_jmeter(root, output, windows))
    if args.include_jmeter_endpoints:
        manifest_rows.extend(convert_jmeter_endpoints(root, output, windows))
    manifest_rows.extend(
        convert_prometheus(root, output, windows, args.prom_label_scope)
    )
    write_manifest(output, manifest_rows)

    print(f"Fault windows: {len(windows)}")
    print(f"KPI files: {len(manifest_rows)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
