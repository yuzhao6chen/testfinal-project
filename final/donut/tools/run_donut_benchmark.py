"""Run Donut on a directory of KPI CSV files with range-based evaluation.

This script is intended for fair comparison with the SR-CNN benchmark in the
``anomalydetector`` directory.  It reads flat CSV files with:

    timestamp,value,label

For each file it trains one Donut model, predicts anomaly scores for the same
series, and evaluates with the same segment/delay rule used by SR-CNN:
if a continuous anomaly segment is detected within ``delay`` points from the
segment start, the whole segment is counted as detected.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run Donut benchmark.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--x-dims", type=int, default=128)
    parser.add_argument("--z-dims", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--valid-step-freq", type=int, default=100)
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mask-labels",
        action="store_true",
        help="Mask known labeled anomaly points during Donut training.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index in the sorted file list to start from.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have score CSVs in the output directory.",
    )
    return parser.parse_args()


def read_kpi_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    by_timestamp: Dict[int, List[float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            timestamp = int(float(row["timestamp"]))
            value = float(row["value"])
            label = int(row["label"])
            # Prometheus files exported in chunks can repeat boundary samples.
            # Match SR-CNN's tolerant evaluation by reducing them to one point:
            # average duplicate values and keep label=1 if any duplicate is 1.
            if timestamp not in by_timestamp:
                by_timestamp[timestamp] = [value, 1.0, float(label)]
            else:
                by_timestamp[timestamp][0] += value
                by_timestamp[timestamp][1] += 1.0
                by_timestamp[timestamp][2] = max(by_timestamp[timestamp][2], float(label))

    duplicate_rows = sum(int(item[1] - 1) for item in by_timestamp.values())
    timestamps, values, labels = [], [], []
    for timestamp, (value_sum, value_count, label) in sorted(by_timestamp.items()):
        timestamps.append(timestamp)
        values.append(value_sum / value_count)
        labels.append(int(label))
    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        duplicate_rows,
    )


def range_adjust(labels: Sequence[int], preds: Sequence[int], delay: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    preds = np.asarray(preds, dtype=np.int32)
    adjusted = np.array(preds, copy=True)
    if len(labels) == 0:
        return adjusted

    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = [0] + splits.tolist()
    ends = splits.tolist() + [len(labels)]
    for start, end in zip(starts, ends):
        if labels[start] != 1:
            continue
        early_end = min(start + delay + 1, end)
        if np.any(preds[start:early_end] == 1):
            adjusted[start:end] = 1
        else:
            adjusted[start:end] = 0
    return adjusted


def metrics(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(preds, dtype=np.int32)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def choose_best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    delay: int,
) -> Tuple[float, Dict[str, float]]:
    if len(scores) == 0:
        return 0.0, metrics(labels, np.zeros_like(labels))

    percentiles = np.linspace(1.0, 99.0, 99)
    candidates = sorted(set(float(np.percentile(scores, p)) for p in percentiles))
    candidates += [float(np.min(scores)) - 1e-8, float(np.max(scores)) + 1e-8]
    best_threshold = candidates[0]
    best_metrics = metrics(labels, range_adjust(labels, scores >= best_threshold, delay))
    for threshold in candidates[1:]:
        adjusted = range_adjust(labels, scores >= threshold, delay)
        current = metrics(labels, adjusted)
        if current["f1"] > best_metrics["f1"]:
            best_metrics = current
            best_threshold = threshold
    return best_threshold, best_metrics


def write_scores(
    path: Path,
    timestamps: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    missing: np.ndarray,
    scores: np.ndarray,
    fixed_threshold: float,
    best_threshold: float,
    delay: int,
) -> None:
    fixed_pred = (scores >= fixed_threshold).astype(np.int32)
    best_pred = (scores >= best_threshold).astype(np.int32)
    best_range_pred = range_adjust(labels, best_pred, delay)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "timestamp",
                "value",
                "label",
                "missing",
                "anomaly_score",
                "fixed_threshold",
                "fixed_pred",
                "best_threshold",
                "best_pred",
                "best_range_pred",
            )
        )
        for row in zip(
            timestamps,
            values,
            labels,
            missing,
            scores,
            fixed_pred,
            best_pred,
            best_range_pred,
        ):
            writer.writerow((*row[:5], fixed_threshold, row[5], best_threshold, row[6], row[7]))


def main() -> None:
    args = parse_args()

    import tensorflow as tf
    from tensorflow import keras as K
    from tfsnippet.modules import Sequential

    from donut import Donut, DonutPredictor, DonutTrainer
    from donut import complete_timestamp, standardize_kpi

    np.random.seed(args.seed)

    all_files = sorted(args.data_dir.glob(args.glob))
    files = all_files[args.start_index - 1 :]
    if args.max_files is not None:
        files = files[: args.max_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores").mkdir(parents=True, exist_ok=True)

    file_summaries = []
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_by_file = {}
            for item in previous.get("files", []):
                previous_by_file[item.get("file")] = item
            file_summaries = list(previous_by_file.values())
        except Exception:
            file_summaries = []

    summarized_by_file = {item.get("file"): item for item in file_summaries}
    aggregate_labels_best = []
    aggregate_preds_best = []
    aggregate_labels_fixed = []
    aggregate_preds_fixed = []

    def read_scored_labels_for_skipped_file(item: Dict[str, object]) -> List[int]:
        path = Path(str(item.get("file")))
        try:
            timestamps, raw_values, labels, _ = read_kpi_csv(path)
            _, _, arrays = complete_timestamp(timestamps, (raw_values, labels))
            _, labels = arrays
            if len(labels) <= args.x_dims:
                return []
            return labels[args.x_dims - 1 :].astype(np.int32).tolist()
        except Exception:
            return []

    def rebuild_aggregates_from_scores() -> Tuple[
        List[int], List[int], List[int], List[int]
    ]:
        aggregate_labels_fixed.clear()
        aggregate_preds_fixed.clear()
        aggregate_labels_best.clear()
        aggregate_preds_best.clear()
        full_labels_fixed: List[int] = []
        full_preds_fixed: List[int] = []
        full_labels_best: List[int] = []
        full_preds_best: List[int] = []
        for item in summarized_by_file.values():
            if item.get("status") == "skipped":
                labels = read_scored_labels_for_skipped_file(item)
                zeros = [0] * len(labels)
                full_labels_fixed.extend(labels)
                full_preds_fixed.extend(zeros)
                full_labels_best.extend(labels)
                full_preds_best.extend(zeros)
                continue
            if item.get("status") != "ok":
                continue
            score_path = item.get("scores_path")
            if not score_path or not Path(score_path).exists():
                continue
            labels, fixed_preds, best_preds = [], [], []
            with Path(score_path).open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    labels.append(int(row["label"]))
                    fixed_preds.append(int(row["fixed_pred"]))
                    best_preds.append(int(row["best_pred"]))
            aggregate_labels_fixed.extend(labels)
            aggregate_preds_fixed.extend(range_adjust(labels, fixed_preds, args.delay).tolist())
            aggregate_labels_best.extend(labels)
            aggregate_preds_best.extend(range_adjust(labels, best_preds, args.delay).tolist())
            full_labels_fixed.extend(labels)
            full_preds_fixed.extend(range_adjust(labels, fixed_preds, args.delay).tolist())
            full_labels_best.extend(labels)
            full_preds_best.extend(range_adjust(labels, best_preds, args.delay).tolist())
        return full_labels_fixed, full_preds_fixed, full_labels_best, full_preds_best

    def upsert_summary(item: Dict[str, object]) -> None:
        summarized_by_file[item.get("file")] = item
        file_summaries[:] = list(summarized_by_file.values())

    def write_summary() -> None:
        (
            full_labels_fixed,
            full_preds_fixed,
            full_labels_best,
            full_preds_best,
        ) = rebuild_aggregates_from_scores()
        ok_items = [
            item for item in summarized_by_file.values()
            if item.get("status") == "ok"
        ]
        skipped_items = [
            item for item in summarized_by_file.values()
            if item.get("status") == "skipped"
        ]
        ok_file_names = sorted(Path(str(item["file"])).name for item in ok_items)
        skipped_file_names = sorted(Path(str(item["file"])).name for item in skipped_items)

        (args.output_dir / "ok_files.txt").write_text(
            "\n".join(ok_file_names) + ("\n" if ok_file_names else ""),
            encoding="utf-8",
        )
        (args.output_dir / "skipped_files.txt").write_text(
            "\n".join(skipped_file_names) + ("\n" if skipped_file_names else ""),
            encoding="utf-8",
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_dir": str(args.data_dir),
                    "x_dims": args.x_dims,
                    "z_dims": args.z_dims,
                    "epochs": args.epochs,
                    "delay": args.delay,
                    "threshold_percentile": args.threshold_percentile,
                    "mask_labels": bool(args.mask_labels),
                    "files_total": len(all_files),
                    "files_ok": sum(
                        1 for item in summarized_by_file.values()
                        if item["status"] == "ok"
                    ),
                    "files_skipped": sum(
                        1 for item in summarized_by_file.values()
                        if item["status"] == "skipped"
                    ),
                    "ok_file_names": ok_file_names,
                    "skipped_file_names": skipped_file_names,
                    "aggregate_fixed_metrics": metrics(
                        aggregate_labels_fixed, aggregate_preds_fixed
                    ),
                    "aggregate_best_metrics": metrics(
                        aggregate_labels_best, aggregate_preds_best
                    ),
                    "aggregate_fixed_metrics_on_ok_files": metrics(
                        aggregate_labels_fixed, aggregate_preds_fixed
                    ),
                    "aggregate_best_metrics_on_ok_files": metrics(
                        aggregate_labels_best, aggregate_preds_best
                    ),
                    "aggregate_fixed_metrics_skipped_as_normal": metrics(
                        full_labels_fixed, full_preds_fixed
                    ),
                    "aggregate_best_metrics_skipped_as_normal": metrics(
                        full_labels_best, full_preds_best
                    ),
                    "files": list(summarized_by_file.values()),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # Normalize old interrupted/resumed summaries before doing more work.
    write_summary()

    for local_index, path in enumerate(files, start=1):
        index = args.start_index + local_index - 1
        score_path = args.output_dir / "scores" / f"{path.stem}_scores.csv"
        previous_item = summarized_by_file.get(str(path))
        if args.skip_existing and previous_item is not None:
            if previous_item.get("status") == "skipped":
                print(f"[{index}/{len(all_files)}] {path} (already skipped)")
                continue
            if previous_item.get("status") == "ok" and score_path.exists():
                print(f"[{index}/{len(all_files)}] {path} (already done)")
                continue

        print(f"[{index}/{len(all_files)}] {path}")
        timestamps, raw_values, labels, duplicate_rows = read_kpi_csv(path)
        try:
            timestamps, missing, arrays = complete_timestamp(timestamps, (raw_values, labels))
            raw_values, labels = arrays

            if len(raw_values) <= args.x_dims:
                raise ValueError("series shorter than x_dims")

            excludes = np.logical_or(labels, missing) if args.mask_labels else missing
            normal_values = raw_values[np.logical_not(excludes)]
            if len(normal_values) == 0:
                raise ValueError("no non-excluded points for standardization")
            mean = float(normal_values.mean())
            std = float(normal_values.std())
            if not np.isfinite(std) or std <= 1e-8:
                raise ValueError("near-zero std")
            std_values, _, _ = standardize_kpi(raw_values, mean=mean, std=std)

            fit_labels = labels if args.mask_labels else np.zeros_like(labels)

            tf.reset_default_graph()
            tf.set_random_seed(args.seed)
            with tf.variable_scope("model") as model_vs:
                model = Donut(
                    h_for_p_x=Sequential(
                        [
                            K.layers.Dense(args.hidden, activation=tf.nn.relu),
                            K.layers.Dense(args.hidden, activation=tf.nn.relu),
                        ]
                    ),
                    h_for_q_z=Sequential(
                        [
                            K.layers.Dense(args.hidden, activation=tf.nn.relu),
                            K.layers.Dense(args.hidden, activation=tf.nn.relu),
                        ]
                    ),
                    x_dims=args.x_dims,
                    z_dims=args.z_dims,
                )

            trainer = DonutTrainer(
                model=model,
                model_vs=model_vs,
                max_epoch=args.epochs,
                batch_size=args.batch_size,
                valid_step_freq=args.valid_step_freq,
            )
            predictor = DonutPredictor(model)

            with tf.Session().as_default():
                trainer.fit(
                    values=std_values,
                    labels=fit_labels,
                    missing=missing,
                    mean=mean,
                    std=std,
                    excludes=excludes,
                )
                log_prob = predictor.get_score(std_values, missing)

            scores = -log_prob
            offset = args.x_dims - 1
            scored_timestamps = timestamps[offset:]
            scored_values = raw_values[offset:]
            scored_labels = labels[offset:]
            scored_missing = missing[offset:]

            fixed_threshold = float(np.percentile(scores, args.threshold_percentile))
            fixed_pred = (scores >= fixed_threshold).astype(np.int32)
            fixed_range_pred = range_adjust(scored_labels, fixed_pred, args.delay)
            fixed_metrics = metrics(scored_labels, fixed_range_pred)

            best_threshold, best_metrics = choose_best_threshold(
                scored_labels, scores, args.delay
            )
            best_pred = (scores >= best_threshold).astype(np.int32)
            best_range_pred = range_adjust(scored_labels, best_pred, args.delay)

            write_scores(
                score_path,
                scored_timestamps,
                scored_values,
                scored_labels,
                scored_missing,
                scores,
                fixed_threshold,
                best_threshold,
                args.delay,
            )

            aggregate_labels_fixed.extend(scored_labels.tolist())
            aggregate_preds_fixed.extend(fixed_range_pred.tolist())
            aggregate_labels_best.extend(scored_labels.tolist())
            aggregate_preds_best.extend(best_range_pred.tolist())

            upsert_summary(
                {
                    "file": str(path),
                    "status": "ok",
                    "points": int(len(raw_values)),
                    "scored_points": int(len(scores)),
                    "label_points": int(np.sum(labels)),
                    "scored_label_points": int(np.sum(scored_labels)),
                    "merged_duplicate_rows": int(duplicate_rows),
                    "fixed_threshold": fixed_threshold,
                    "fixed_metrics": fixed_metrics,
                    "best_threshold": best_threshold,
                    "best_metrics": best_metrics,
                    "scores_path": str(score_path),
                }
            )
        except Exception as exc:
            print(f"  skipped: {exc}")
            upsert_summary(
                {
                    "file": str(path),
                    "status": "skipped",
                    "reason": repr(exc),
                }
            )

        write_summary()

    print(f"summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
