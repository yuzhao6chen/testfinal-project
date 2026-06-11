"""Train and evaluate Donut on one Donut-format KPI CSV.

Input CSV schema:

    timestamp,value,label

The script is intentionally small and follows the API usage from the upstream
Donut README.  It writes scored test points to ``scores.csv`` and a compact
``summary.json`` with basic metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Donut on one KPI CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("online_boutique_donut_data/jmeter/jmeter__p95_elapsed_ms.csv"),
        help="Donut-format KPI CSV with timestamp,value,label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("donut_runs/p95_elapsed_ms"),
        help="Directory for scores.csv and summary.json.",
    )
    parser.add_argument("--train-portion", type=float, default=0.7)
    parser.add_argument("--x-dims", type=int, default=120)
    parser.add_argument("--z-dims", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--valid-step-freq", type=int, default=100)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mask-train-labels",
        action="store_true",
        help=(
            "Mask labeled anomaly points during training. Leave this off for a "
            "stricter unsupervised run; turn it on to follow the upstream README "
            "style when labels are trusted."
        ),
    )
    return parser.parse_args()


def read_kpi_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    values = []
    labels = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            timestamps.append(int(float(row["timestamp"])))
            values.append(float(row["value"]))
            labels.append(int(row["label"]))
    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def write_scores(
    path: Path,
    timestamps: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    missing: np.ndarray,
    reconstruction_log_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    anomaly_score = -reconstruction_log_prob
    pred = (anomaly_score >= threshold).astype(np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "timestamp",
                "value",
                "label",
                "missing",
                "reconstruction_log_prob",
                "anomaly_score",
                "threshold",
                "pred",
            )
        )
        for row in zip(
            timestamps,
            values,
            labels,
            missing,
            reconstruction_log_prob,
            anomaly_score,
            pred,
        ):
            writer.writerow((*row[:6], threshold, row[6]))

    return binary_metrics(labels, pred)


def binary_metrics(labels: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int32)
    pred = pred.astype(np.int32)
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tn = int(np.sum((labels == 0) & (pred == 0)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / float(precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()

    # These imports require the TensorFlow 1.x Donut environment.
    import tensorflow as tf
    from tensorflow import keras as K
    from tfsnippet.modules import Sequential

    from donut import Donut, DonutPredictor, DonutTrainer
    from donut import complete_timestamp, standardize_kpi

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    timestamps, values, labels = read_kpi_csv(args.input)
    timestamps, missing, arrays = complete_timestamp(timestamps, (values, labels))
    values, labels = arrays

    train_n = int(len(values) * args.train_portion)
    if train_n <= args.x_dims or len(values) - train_n <= args.x_dims:
        raise ValueError("Train/test split is too small for --x-dims")

    train_values_raw = values[:train_n]
    test_values_raw = values[train_n:]
    train_labels = labels[:train_n]
    test_labels = labels[train_n:]
    train_missing = missing[:train_n]
    test_missing = missing[train_n:]
    test_timestamps = timestamps[train_n:]

    fit_labels = train_labels if args.mask_train_labels else np.zeros_like(train_labels)
    excludes = np.logical_or(fit_labels, train_missing)
    train_values, mean, std = standardize_kpi(train_values_raw, excludes=excludes)
    if not np.isfinite(std) or std <= 1e-8:
        raise ValueError("Training series has near-zero std; choose another KPI.")
    test_values, _, _ = standardize_kpi(test_values_raw, mean=mean, std=std)

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
            values=train_values,
            labels=fit_labels,
            missing=train_missing,
            mean=mean,
            std=std,
            excludes=excludes,
        )
        train_log_prob = predictor.get_score(train_values, train_missing)
        test_log_prob = predictor.get_score(test_values, test_missing)

    threshold = float(np.percentile(-train_log_prob, args.threshold_percentile))
    offset = args.x_dims - 1
    scored_timestamps = test_timestamps[offset:]
    scored_values = test_values_raw[offset:]
    scored_labels = test_labels[offset:]
    scored_missing = test_missing[offset:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_scores(
        args.output_dir / "scores.csv",
        scored_timestamps,
        scored_values,
        scored_labels,
        scored_missing,
        test_log_prob,
        threshold,
    )

    summary = {
        "input": str(args.input),
        "train_points": int(len(train_values)),
        "test_points": int(len(test_values)),
        "scored_test_points": int(len(test_log_prob)),
        "train_label_points": int(np.sum(train_labels)),
        "test_label_points": int(np.sum(test_labels)),
        "scored_test_label_points": int(np.sum(scored_labels)),
        "train_missing_points": int(np.sum(train_missing)),
        "test_missing_points": int(np.sum(test_missing)),
        "x_dims": args.x_dims,
        "z_dims": args.z_dims,
        "epochs": args.epochs,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "mask_train_labels": bool(args.mask_train_labels),
        "metrics": metrics,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
