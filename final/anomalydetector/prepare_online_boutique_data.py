import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


JMETER_VALUE_COLUMNS = (
    "request_count",
    "success_count",
    "error_count",
    "error_rate",
    "avg_elapsed_ms",
    "p95_elapsed_ms",
    "p99_elapsed_ms",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the collected Online Boutique data into SR-CNN CSV inputs."
    )
    parser.add_argument(
        "--final-data",
        default="../\u6700\u7ec8\u6570\u636e",
        help="Path to the final collected data directory.",
    )
    parser.add_argument(
        "--train-output",
        default="online_boutique_train",
        help="Nested timestamp,value dataset for srcnn/generate_data.py.",
    )
    parser.add_argument(
        "--test-output",
        default="test_kpi",
        help="Flat timestamp,value,label dataset for srcnn/evalue.py --data test_kpi.",
    )
    parser.add_argument(
        "--label-end",
        choices=("delete", "recovered"),
        default="recovered",
        help="End labels at fault_delete_done or recovered_confirmed.",
    )
    return parser.parse_args()


def resolve_path(path_text, base_dir):
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_time(value):
    return datetime.fromisoformat(value).timestamp()


def sanitize(value, fallback="series"):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value[:120] or fallback


def unique_name(used, name):
    if name not in used:
        used.add(name)
        return name
    idx = 2
    while f"{name}_{idx}" in used:
        idx += 1
    name = f"{name}_{idx}"
    used.add(name)
    return name


def read_fault_intervals(log_path, label_end):
    phases = {}
    with log_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            event = json.loads(line)
            phase_id = event.get("phase_id")
            if not phase_id:
                continue
            phase = phases.setdefault(
                phase_id,
                {
                    "phase_id": phase_id,
                    "fault_type": event.get("fault_type"),
                    "target_service": event.get("target_service"),
                },
            )
            if event["event"] == "fault_apply_start":
                phase["start"] = parse_time(event["time"])
                phase["start_iso"] = event["time"]
            elif event["event"] == "fault_delete_done":
                phase["delete"] = parse_time(event["time"])
                phase["delete_iso"] = event["time"]
            elif event["event"] == "recovered_confirmed":
                phase["recovered"] = parse_time(event["time"])
                phase["recovered_iso"] = event["time"]

    intervals = []
    end_key = "delete" if label_end == "delete" else "recovered"
    for phase in phases.values():
        if "start" not in phase or end_key not in phase:
            continue
        intervals.append(
            {
                "phase_id": phase["phase_id"],
                "fault_type": phase.get("fault_type"),
                "target_service": phase.get("target_service"),
                "start": phase["start"],
                "end": phase[end_key],
                "start_iso": phase["start_iso"],
                "end_iso": phase[f"{end_key}_iso"],
            }
        )
    intervals.sort(key=lambda item: item["start"])
    return intervals


def is_labeled(ts, intervals):
    return int(any(item["start"] <= ts <= item["end"] for item in intervals))


def target_service_for_labels(labels):
    if "deployment" in labels:
        return labels["deployment"]
    pod = labels.get("pod")
    if pod:
        return pod
    return None


def matching_intervals(metric_id, labels, intervals):
    if metric_id.startswith("node_"):
        return intervals

    target = target_service_for_labels(labels)
    if not target:
        return intervals

    matched = []
    for interval in intervals:
        service = interval.get("target_service")
        if not service:
            continue
        if target == service or target.startswith(service + "-"):
            matched.append(interval)
    return matched


def write_series(path, rows, include_label):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "value", "label"] if include_label else ["timestamp", "value"]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def read_jmeter_rows(final_data):
    summary_dir = final_data / "summary"
    matches = list(summary_dir.glob("jmeter_5s*_kpi.csv"))
    if not matches:
        raise FileNotFoundError("Could not find summary/jmeter_5s*_kpi.csv")
    with matches[0].open("r", encoding="utf-8-sig", newline="") as fin:
        return matches[0], list(csv.DictReader(fin))


def convert_jmeter(final_data, train_output, test_output, intervals, used_test_names):
    jmeter_path, rows = read_jmeter_rows(final_data)
    train_dir = train_output / "jmeter"
    count = 0
    for column in JMETER_VALUE_COLUMNS:
        if column not in rows[0]:
            continue
        train_rows = []
        test_rows = []
        for row in rows:
            timestamp = int(float(row["bin_start"]))
            value = row[column]
            train_rows.append([timestamp, value])
            test_rows.append([timestamp, value, is_labeled(timestamp, intervals)])
        write_series(train_dir / f"jmeter_{column}.csv", train_rows, include_label=False)
        test_name = unique_name(used_test_names, f"jmeter_{column}") + ".csv"
        write_series(test_output / test_name, test_rows, include_label=True)
        count += 1
    return {"source": str(jmeter_path), "series": count, "rows_per_series": len(rows)}


def series_display_name(metric_id, labels, index):
    if "deployment" in labels:
        base = labels["deployment"]
    elif "pod" in labels:
        base = labels["pod"]
    elif "instance" in labels:
        base = labels["instance"]
    else:
        base = f"series_{index:03d}"

    container = labels.get("container")
    if container and container not in {"main", base}:
        base = f"{base}_{container}"
    return sanitize(f"{metric_id}_{base}", fallback=f"{metric_id}_{index:03d}")


def convert_prometheus(final_data, train_output, test_output, intervals, used_test_names):
    raw_dir = final_data / "raw_prometheus"
    summary = {}

    for metric_path in sorted(raw_dir.glob("*.csv")):
        groups = defaultdict(list)
        labels_by_key = {}
        with metric_path.open("r", encoding="utf-8-sig", newline="") as fin:
            for row in csv.DictReader(fin):
                labels = json.loads(row["labels_json"])
                key = json.dumps(labels, sort_keys=True, ensure_ascii=True)
                labels_by_key[key] = labels
                groups[key].append((float(row["timestamp"]), row["value"]))

        metric_id = metric_path.stem
        metric_train_dir = train_output / metric_id
        lengths = []
        used_train_names = set()
        for index, (key, points) in enumerate(sorted(groups.items()), start=1):
            labels = labels_by_key[key]
            name = series_display_name(metric_id, labels, index)
            train_name = unique_name(used_train_names, name)
            test_name = unique_name(used_test_names, name)
            points.sort(key=lambda item: item[0])

            train_rows = [[int(round(ts)), value] for ts, value in points]
            label_intervals = matching_intervals(metric_id, labels, intervals)
            test_rows = [
                [int(round(ts)), value, is_labeled(ts, label_intervals)]
                for ts, value in points
            ]
            write_series(metric_train_dir / f"{train_name}.csv", train_rows, include_label=False)
            write_series(test_output / f"{test_name}.csv", test_rows, include_label=True)
            lengths.append(len(points))

        summary[metric_id] = {
            "series": len(groups),
            "rows": sum(lengths),
            "min_rows_per_series": min(lengths) if lengths else 0,
            "max_rows_per_series": max(lengths) if lengths else 0,
        }
    return summary


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    final_data = resolve_path(args.final_data, script_dir)
    train_output = resolve_path(args.train_output, script_dir)
    test_output = resolve_path(args.test_output, script_dir)

    if not final_data.exists():
        raise FileNotFoundError(final_data)
    train_output.mkdir(parents=True, exist_ok=True)
    test_output.mkdir(parents=True, exist_ok=True)

    intervals = read_fault_intervals(final_data / "logs" / "fault_events.jsonl", args.label_end)
    used_test_names = set()
    jmeter_summary = convert_jmeter(final_data, train_output, test_output, intervals, used_test_names)
    prometheus_summary = convert_prometheus(final_data, train_output, test_output, intervals, used_test_names)

    summary = {
        "final_data": str(final_data),
        "train_output": str(train_output),
        "test_output": str(test_output),
        "label_end": args.label_end,
        "fault_intervals": intervals,
        "jmeter": jmeter_summary,
        "prometheus": prometheus_summary,
        "notes": [
            "Training files are timestamp,value and are organized as dataset/subdir/*.csv for srcnn/generate_data.py.",
            "Test files are timestamp,value,label and are flat for srcnn/evalue.py --data test_kpi.",
            "JMeter and node metrics use global fault-window labels.",
            "Pod and deployment metrics use labels only for matching target services when possible.",
        ],
    }
    summary_path = script_dir / "online_boutique_data_summary.json"
    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)

    print(f"fault intervals: {len(intervals)}")
    print(f"jmeter series: {jmeter_summary['series']}")
    print(f"prometheus metric groups: {len(prometheus_summary)}")
    print(f"train output: {train_output}")
    print(f"test output: {test_output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
