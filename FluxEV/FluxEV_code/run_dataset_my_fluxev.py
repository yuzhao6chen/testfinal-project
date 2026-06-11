import argparse
import contextlib
import csv
import io
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from eval_methods import adjust_predicts


ALGORITHM_NAME = "FluxEV"
DEFAULT_INTERVAL_SECONDS = 5
RECALL_F1_TOLERANCE = 0.005


@dataclass(frozen=True)
class ParamSpec:
    params_id: str
    estimator: str
    smoothing: int
    period: int
    s_w: int
    p_w: int
    half_d_w: int
    q: float
    delay: int


@dataclass
class PreparedSplit:
    timestamp: np.ndarray
    value: np.ndarray
    label: np.ndarray
    missing: np.ndarray
    original_rows: int
    completed_rows: int
    duplicate_rows: int


@dataclass
class SeriesData:
    dataset: str
    metric_name: str
    train_path: Path
    test_path: Path
    train: PreparedSplit
    test: PreparedSplit

    @property
    def has_test_anomaly(self) -> bool:
        return bool(np.any(self.test.label == 1))


@dataclass
class PredictionResult:
    dataset: str
    metric_name: str
    params_id: str
    timestamp: np.ndarray
    value: np.ndarray
    label: np.ndarray
    missing: np.ndarray
    raw_pred: np.ndarray
    adjusted_pred: np.ndarray
    error: str = ""


@dataclass
class SelectionChoice:
    dataset_key: str
    dataset_label: str
    selected_row: Dict[str, object]
    max_f1_row: Dict[str, object]
    near_f1_max_recall_row: Dict[str, object]
    max_recall_row: Dict[str, object]
    reason: str
    baseline_recall: Optional[float]
    recall_improved_vs_baseline: Optional[bool]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    code_output_root = script_dir.parent
    fluxev_root = code_output_root.parent if code_output_root.name == "code_output" else code_output_root
    default_output_root = (
        code_output_root / "outputs_dataset_my_final"
        if code_output_root.name == "code_output"
        else fluxev_root / "outputs_dataset_my_final"
    )
    parser = argparse.ArgumentParser(
        description="Run FluxEV on dataset_my and produce Donut/SR-CNN style tables."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=fluxev_root / "dataset_my_final",
        help="Path to FluxEV/dataset_my_final.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Directory for timestamped final-dataset experiment outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "grid", "plots-only"],
        default="grid",
        help="smoke runs a small parameter set; grid runs the configured search.",
    )
    parser.add_argument(
        "--grid-preset",
        choices=["default", "recall_fine"],
        default="default",
        help="Parameter grid preset. recall_fine runs the 195-set recall search.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["max_f1", "near_f1_max_recall"],
        default=None,
        help=(
            "Best-parameter selection strategy. Defaults to near_f1_max_recall "
            "for recall_fine and max_f1 otherwise."
        ),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help=(
            "Optional final_report_table.csv used as the current clean baseline "
            "for recall-improvement gating."
        ),
    )
    parser.add_argument(
        "--existing-output",
        type=Path,
        default=None,
        help="Existing output directory used by plots-only mode.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Sampling interval used to complete the timestamp grid.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip writing final best prediction CSV files.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional output run id. Defaults to current time.",
    )
    return parser.parse_args()


def make_param(smoothing: int, period: int, s_w: int, q: float) -> ParamSpec:
    period_part = "NA" if smoothing == 1 else str(period)
    return ParamSpec(
        params_id=f"S{smoothing}P{period_part}_SW{s_w}_Q{q:g}",
        estimator="MOM",
        smoothing=smoothing,
        period=period,
        s_w=s_w,
        p_w=5,
        half_d_w=2,
        q=q,
        delay=7,
    )


def build_param_grid(mode: str, grid_preset: str = "default") -> List[ParamSpec]:
    if mode == "smoke":
        if grid_preset == "recall_fine":
            return [
                make_param(1, 0, 5, 0.003),
                make_param(1, 0, 20, 0.005),
                make_param(1, 0, 5, 0.001),
            ]
        return [
            make_param(2, 60, 10, 0.003),
        ]

    smoothing_periods = [(1, 0), (2, 60), (2, 120)]
    params: List[ParamSpec] = []
    if grid_preset == "recall_fine":
        fine_q_values = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
        fine_s_w_values = [3, 4, 5, 6, 8, 10, 12]
        extended_q_values = [0.006, 0.008, 0.01, 0.015]
        extended_s_w_values = [5, 10, 20, 30]
        for smoothing, period in smoothing_periods:
            for s_w in fine_s_w_values:
                for q in fine_q_values:
                    params.append(make_param(smoothing, period, s_w, q))
            for s_w in extended_s_w_values:
                for q in extended_q_values:
                    params.append(make_param(smoothing, period, s_w, q))
    else:
        q_values = [0.001, 0.003, 0.005]
        s_w_values = [5, 10, 20]
        for smoothing, period in smoothing_periods:
            for s_w in s_w_values:
                for q in q_values:
                    params.append(make_param(smoothing, period, s_w, q))
    return params


def collect_csv_by_basename(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*.csv")):
        if path.name in mapping:
            duplicates.setdefault(path.name, [mapping[path.name]]).append(path)
        else:
            mapping[path.name] = path
    if duplicates:
        details = "; ".join(f"{name}: {paths}" for name, paths in duplicates.items())
        raise ValueError(f"Duplicate CSV basenames under {root}: {details}")
    return mapping


def require_columns(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_and_complete_split(
    path: Path,
    has_label: bool,
    interval_seconds: int,
) -> PreparedSplit:
    df = pd.read_csv(path)
    require_columns(df, path, ["timestamp", "value"])
    if has_label:
        require_columns(df, path, ["label"])
    else:
        df["label"] = 0

    original_rows = len(df)
    df = df[["timestamp", "value", "label"]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["label"] = (df["label"] > 0).astype(np.int64)
    df["_order"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(["timestamp", "_order"], kind="mergesort")

    duplicate_rows = int(df.duplicated(subset=["timestamp"], keep="first").sum())
    df = (
        df.groupby("timestamp", sort=True)
        .agg(value=("value", "last"), label=("label", "max"))
        .reset_index()
    )

    if df.empty:
        raise ValueError(f"{path} has no usable timestamp rows")

    start = int(df["timestamp"].iloc[0])
    end = int(df["timestamp"].iloc[-1])
    full_timestamp = np.arange(start, end + interval_seconds, interval_seconds, dtype=np.int64)
    completed = pd.DataFrame({"timestamp": full_timestamp}).merge(df, on="timestamp", how="left")
    missing = completed["value"].isna().to_numpy(dtype=np.int64)
    completed["label"] = completed["label"].fillna(0).astype(np.int64)
    completed["value"] = completed["value"].interpolate(method="linear", limit_direction="both")
    completed["value"] = completed["value"].fillna(0.0)

    value = completed["value"].to_numpy(dtype=np.float64)
    label = (completed["label"].to_numpy(dtype=np.int64) > 0).astype(np.int64)
    return PreparedSplit(
        timestamp=full_timestamp,
        value=value,
        label=label,
        missing=missing,
        original_rows=original_rows,
        completed_rows=len(completed),
        duplicate_rows=duplicate_rows,
    )


def load_dataset(dataset_root: Path, interval_seconds: int) -> List[SeriesData]:
    specs = [
        ("jmeter", dataset_root / "train_jmeter", dataset_root / "test_jmeter"),
        ("prometheus", dataset_root / "train_prometheus", dataset_root / "test_prometheus"),
    ]
    all_series: List[SeriesData] = []
    for dataset, train_root, test_root in specs:
        train_files = collect_csv_by_basename(train_root)
        test_files = collect_csv_by_basename(test_root)
        missing_train = sorted(set(test_files) - set(train_files))
        missing_test = sorted(set(train_files) - set(test_files))
        if missing_train or missing_test:
            raise ValueError(
                f"{dataset} train/test mismatch. "
                f"missing_train={missing_train}, missing_test={missing_test}"
            )

        for metric_name in sorted(test_files):
            train = read_and_complete_split(train_files[metric_name], False, interval_seconds)
            test = read_and_complete_split(test_files[metric_name], True, interval_seconds)
            all_series.append(
                SeriesData(
                    dataset=dataset,
                    metric_name=metric_name,
                    train_path=train_files[metric_name],
                    test_path=test_files[metric_name],
                    train=train,
                    test=test,
                )
            )
    return all_series


def validate_dataset(series_list: Sequence[SeriesData]) -> Dict[str, object]:
    by_dataset: Dict[str, List[SeriesData]] = {}
    for series in series_list:
        by_dataset.setdefault(series.dataset, []).append(series)

    summary: Dict[str, object] = {}
    for dataset, items in sorted(by_dataset.items()):
        summary[dataset] = {
            "files": len(items),
            "valid_with_anomaly": sum(1 for item in items if item.has_test_anomaly),
            "train_completed_rows_min": int(min(item.train.completed_rows for item in items)),
            "train_completed_rows_max": int(max(item.train.completed_rows for item in items)),
            "test_completed_rows_min": int(min(item.test.completed_rows for item in items)),
            "test_completed_rows_max": int(max(item.test.completed_rows for item in items)),
            "train_duplicate_rows": int(sum(item.train.duplicate_rows for item in items)),
            "test_duplicate_rows": int(sum(item.test.duplicate_rows for item in items)),
        }
    for dataset in ["jmeter", "prometheus"]:
        actual = len(by_dataset.get(dataset, []))
        if actual == 0:
            raise ValueError(f"No {dataset} metrics found")
    return summary


def safe_detect(series: SeriesData, params: ParamSpec) -> Tuple[np.ndarray, str]:
    values = np.concatenate([series.train.value, series.test.value]).astype(np.float64)
    train_len = len(series.train.value)
    period = params.period if params.smoothing == 2 else 1
    required_init = params.s_w * 2
    if params.smoothing == 2:
        required_init = params.s_w * 2 + params.half_d_w + period * (params.p_w - 1)
    if train_len <= required_init + 5:
        return np.zeros(len(series.test.value), dtype=np.int64), (
            f"train_len {train_len} too short for required_init {required_init}"
        )
    if not np.all(np.isfinite(values)):
        return np.zeros(len(series.test.value), dtype=np.int64), "non-finite input values"

    try:
        from main import detect as fluxev_detect

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with contextlib.redirect_stdout(io.StringIO()):
                alarms = fluxev_detect(
                    values,
                    train_len=train_len,
                    period=period,
                    smoothing=params.smoothing,
                    s_w=params.s_w,
                    p_w=params.p_w,
                    half_d_w=params.half_d_w,
                    q=params.q,
                    estimator=params.estimator,
                )
        alarms = np.asarray(alarms, dtype=np.int64)
    except Exception as exc:
        return np.zeros(len(series.test.value), dtype=np.int64), repr(exc)

    expected = len(series.test.value)
    if len(alarms) < expected:
        padded = np.zeros(expected, dtype=np.int64)
        padded[: len(alarms)] = alarms
        alarms = padded
    elif len(alarms) > expected:
        alarms = alarms[:expected]
    alarms = (alarms > 0).astype(np.int64)
    alarms[series.test.missing == 1] = 0
    return alarms, ""


def evaluate_series(series: SeriesData, params: ParamSpec) -> PredictionResult:
    raw_pred, error = safe_detect(series, params)
    adjusted = adjust_predicts(raw_pred, series.test.label, delay=params.delay).astype(np.int64)
    adjusted = (adjusted > 0).astype(np.int64)
    adjusted[series.test.missing == 1] = 0
    return PredictionResult(
        dataset=series.dataset,
        metric_name=series.metric_name,
        params_id=params.params_id,
        timestamp=series.test.timestamp,
        value=series.test.value,
        label=series.test.label,
        missing=series.test.missing,
        raw_pred=raw_pred,
        adjusted_pred=adjusted,
        error=error,
    )


def pooled_metrics(results: Iterable[PredictionResult]) -> Dict[str, float]:
    result_list = list(results)
    if not result_list:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0, "predicted": 0}
    y_true = np.concatenate([result.label for result in result_list])
    y_pred = np.concatenate([result.adjusted_pred for result in result_list])
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "support": int(y_true.sum()),
        "predicted": int(y_pred.sum()),
    }


def per_file_metrics(result: PredictionResult) -> Dict[str, object]:
    return {
        "dataset": result.dataset,
        "metric_name": result.metric_name,
        "params_id": result.params_id,
        "f1": float(f1_score(result.label, result.adjusted_pred, zero_division=0)),
        "precision": float(precision_score(result.label, result.adjusted_pred, zero_division=0)),
        "recall": float(recall_score(result.label, result.adjusted_pred, zero_division=0)),
        "support": int(result.label.sum()),
        "predicted": int(result.adjusted_pred.sum()),
        "rows": int(len(result.label)),
        "missing_rows": int(result.missing.sum()),
        "error": result.error,
    }


def run_param(series_list: Sequence[SeriesData], params: ParamSpec) -> List[PredictionResult]:
    return [evaluate_series(series, params) for series in series_list]


def summarize_param(params: ParamSpec, results: Sequence[PredictionResult]) -> List[Dict[str, object]]:
    jmeter = [result for result in results if result.dataset == "jmeter"]
    prometheus_all = [result for result in results if result.dataset == "prometheus"]
    prometheus_valid = [result for result in prometheus_all if np.any(result.label == 1)]
    groups = [
        ("JMeter", jmeter),
        (f"Prometheus 有效 {len(prometheus_valid)} 指标", prometheus_valid),
        (f"Prometheus 全 {len(prometheus_all)} 指标", prometheus_all),
    ]

    rows: List[Dict[str, object]] = []
    for label, group_results in groups:
        metrics = pooled_metrics(group_results)
        rows.append(
            {
                "params_id": params.params_id,
                "dataset": label,
                "algorithm": ALGORITHM_NAME,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "support": metrics["support"],
                "predicted": metrics["predicted"],
                "files": len(group_results),
                "estimator": params.estimator,
                "smoothing": params.smoothing,
                "period": params.period if params.smoothing == 2 else "",
                "s_w": params.s_w,
                "p_w": params.p_w,
                "half_d_w": params.half_d_w,
                "q": params.q,
                "delay": params.delay,
                "errors": sum(1 for result in group_results if result.error),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        return default
    return float(value)


def choose_max_f1_row(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return sorted(
        candidates,
        key=lambda row: (
            row_float(row, "f1"),
            row_float(row, "recall"),
            row_float(row, "precision"),
            -row_float(row, "q"),
        ),
        reverse=True,
    )[0]


def choose_max_recall_row(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return sorted(
        candidates,
        key=lambda row: (
            row_float(row, "recall"),
            row_float(row, "f1"),
            row_float(row, "precision"),
            -row_float(row, "q"),
        ),
        reverse=True,
    )[0]


def choose_near_f1_max_recall_row(
    candidates: Sequence[Dict[str, object]],
    tolerance: float = RECALL_F1_TOLERANCE,
) -> Dict[str, object]:
    max_f1 = row_float(choose_max_f1_row(candidates), "f1")
    near_candidates = [row for row in candidates if row_float(row, "f1") >= max_f1 - tolerance]
    return choose_max_recall_row(near_candidates)


def find_default_baseline_report(fluxev_root: Path) -> Optional[Path]:
    baseline_root = fluxev_root / "outputs_dataset_my_clean"
    if not baseline_root.exists():
        return None
    candidates = sorted(
        baseline_root.glob("*/final_report_table.csv"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for path in candidates:
        if path.parent.name.endswith("_smoke"):
            continue
        return path
    return candidates[0] if candidates else None


def load_baseline_metrics(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    metrics: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        label = str(row["数据集"])
        metrics[label] = {
            "params_id": "baseline",
            "dataset": label,
            "algorithm": str(row.get("算法", ALGORITHM_NAME)),
            "f1": float(row["F1"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "support": "",
            "predicted": "",
            "files": "",
        }
    return metrics


def select_best_params(
    grid_rows: Sequence[Dict[str, object]],
    selection_strategy: str = "max_f1",
    baseline_metrics: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, SelectionChoice]:
    baseline_metrics = baseline_metrics or {}

    def select_for(dataset_key: str, dataset_match) -> SelectionChoice:
        candidates = [row for row in grid_rows if dataset_match(str(row["dataset"]))]
        if not candidates:
            raise ValueError("No candidate rows matched for best parameter selection")

        max_f1_row = choose_max_f1_row(candidates)
        near_row = choose_near_f1_max_recall_row(candidates)
        max_recall_row = choose_max_recall_row(candidates)
        dataset_label = str(max_f1_row["dataset"])
        baseline_row = baseline_metrics.get(dataset_label)
        baseline_recall = (
            row_float(baseline_row, "recall") if baseline_row is not None else None
        )

        if selection_strategy == "near_f1_max_recall":
            improved = (
                None
                if baseline_recall is None
                else row_float(near_row, "recall") > baseline_recall + 1e-12
            )
            if improved is False:
                selected_row = max_f1_row
                reason = "no_recall_improvement_vs_baseline_keep_max_f1"
            else:
                selected_row = near_row
                reason = "near_f1_max_recall"
        else:
            improved = (
                None
                if baseline_recall is None
                else row_float(max_f1_row, "recall") > baseline_recall + 1e-12
            )
            selected_row = max_f1_row
            reason = "max_f1"

        return SelectionChoice(
            dataset_key=dataset_key,
            dataset_label=dataset_label,
            selected_row=selected_row,
            max_f1_row=max_f1_row,
            near_f1_max_recall_row=near_row,
            max_recall_row=max_recall_row,
            reason=reason,
            baseline_recall=baseline_recall,
            recall_improved_vs_baseline=improved,
        )

    return {
        "jmeter": select_for("jmeter", lambda label: label == "JMeter"),
        "prometheus": select_for(
            "prometheus", lambda label: label.startswith("Prometheus 有效 ")
        ),
    }


def selection_best_ids(selection: Dict[str, SelectionChoice]) -> Dict[str, str]:
    return {
        key: str(choice.selected_row["params_id"]) for key, choice in selection.items()
    }


def selection_max_f1_ids(selection: Dict[str, SelectionChoice]) -> Dict[str, str]:
    return {
        key: str(choice.max_f1_row["params_id"]) for key, choice in selection.items()
    }


def build_best_params_rows(selection: Dict[str, SelectionChoice]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset, choice in sorted(selection.items()):
        rows.append(
            {
                "dataset": dataset,
                "best_params_id": choice.selected_row["params_id"],
                "selection_strategy": choice.reason,
                "dataset_label": choice.dataset_label,
                "selected_f1": choice.selected_row["f1"],
                "selected_precision": choice.selected_row["precision"],
                "selected_recall": choice.selected_row["recall"],
                "max_f1": choice.max_f1_row["f1"],
                "near_f1_max_recall": choice.near_f1_max_recall_row["recall"],
                "max_recall": choice.max_recall_row["recall"],
                "baseline_recall": (
                    "" if choice.baseline_recall is None else choice.baseline_recall
                ),
                "recall_improved_vs_baseline": (
                    ""
                    if choice.recall_improved_vs_baseline is None
                    else choice.recall_improved_vs_baseline
                ),
            }
        )
    return rows


def build_recall_tradeoff_rows(
    grid_rows: Sequence[Dict[str, object]],
    selection: Dict[str, SelectionChoice],
    baseline_metrics: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    labels: List[str] = []
    for row in grid_rows:
        label = str(row["dataset"])
        if label not in labels:
            labels.append(label)

    selected_by_label = {
        choice.dataset_label: str(choice.selected_row["params_id"])
        for choice in selection.values()
    }
    final_selection_labels = set(selected_by_label.keys())

    def append_row(
        dataset_label: str,
        source: str,
        row: Dict[str, object],
        candidate_count: int,
        best_f1: float,
    ) -> None:
        params_id = str(row.get("params_id", ""))
        baseline_row = baseline_metrics.get(dataset_label)
        baseline_recall = (
            "" if baseline_row is None else row_float(baseline_row, "recall")
        )
        selected_for_final = (
            dataset_label in final_selection_labels
            and selected_by_label[dataset_label] == params_id
            and source != "baseline"
        )
        recall_improved = (
            ""
            if baseline_row is None or source == "baseline"
            else row_float(row, "recall") > row_float(baseline_row, "recall") + 1e-12
        )
        rows.append(
            {
                "dataset": dataset_label,
                "source": source,
                "params_id": params_id,
                "f1": row.get("f1", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "support": row.get("support", ""),
                "predicted": row.get("predicted", ""),
                "files": row.get("files", ""),
                "candidate_count": candidate_count,
                "best_f1": best_f1,
                "f1_tolerance": RECALL_F1_TOLERANCE,
                "baseline_recall": baseline_recall,
                "recall_improved_vs_baseline": recall_improved,
                "selected_for_final": selected_for_final,
            }
        )

    for dataset_label in labels:
        candidates = [row for row in grid_rows if str(row["dataset"]) == dataset_label]
        max_f1_row = choose_max_f1_row(candidates)
        near_row = choose_near_f1_max_recall_row(candidates)
        max_recall_row = choose_max_recall_row(candidates)
        best_f1 = row_float(max_f1_row, "f1")
        baseline_row = baseline_metrics.get(dataset_label)
        if baseline_row is not None:
            append_row(
                dataset_label,
                "baseline",
                baseline_row,
                len(candidates),
                best_f1,
            )
        append_row(dataset_label, "max_f1", max_f1_row, len(candidates), best_f1)
        append_row(
            dataset_label,
            "near_f1_max_recall",
            near_row,
            len(candidates),
            best_f1,
        )
        append_row(dataset_label, "max_recall", max_recall_row, len(candidates), best_f1)
    return rows


def result_key(result: PredictionResult) -> Tuple[str, str]:
    return result.dataset, result.metric_name


def write_predictions(output_dir: Path, results: Sequence[PredictionResult]) -> None:
    for result in results:
        pred_dir = output_dir / "predictions" / f"{result.dataset}_best"
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"{Path(result.metric_name).stem}_predictions.csv"
        df = pd.DataFrame(
            {
                "timestamp": result.timestamp,
                "value": result.value,
                "label": result.label,
                "raw_pred": result.raw_pred,
                "adjusted_pred": result.adjusted_pred,
                "missing": result.missing,
                "params_id": result.params_id,
            }
        )
        df.to_csv(path, index=False, encoding="utf-8-sig")


def read_best_params(output_dir: Path) -> Dict[str, str]:
    path = output_dir / "best_params.csv"
    df = pd.read_csv(path)
    return {str(row["dataset"]): str(row["best_params_id"]) for _, row in df.iterrows()}


def load_existing_best_predictions(
    output_dir: Path,
    series_list: Sequence[SeriesData],
) -> List[PredictionResult]:
    results: List[PredictionResult] = []
    for series in series_list:
        pred_path = (
            output_dir
            / "predictions"
            / f"{series.dataset}_best"
            / f"{Path(series.metric_name).stem}_predictions.csv"
        )
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        df = pd.read_csv(pred_path)
        results.append(
            PredictionResult(
                dataset=series.dataset,
                metric_name=series.metric_name,
                params_id=str(df["params_id"].iloc[0]),
                timestamp=df["timestamp"].to_numpy(dtype=np.int64),
                value=df["value"].to_numpy(dtype=np.float64),
                label=df["label"].to_numpy(dtype=np.int64),
                missing=df["missing"].to_numpy(dtype=np.int64),
                raw_pred=df["raw_pred"].to_numpy(dtype=np.int64),
                adjusted_pred=df["adjusted_pred"].to_numpy(dtype=np.int64),
            )
        )
    return results


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    indexes = np.where(mask > 0)[0]
    if indexes.size == 0:
        return []
    splits = np.where(np.diff(indexes) > 1)[0] + 1
    groups = np.split(indexes, splits)
    return [(int(group[0]), int(group[-1])) for group in groups if len(group) > 0]


def build_segment_delay_rows(
    results: Sequence[PredictionResult],
    params_by_id: Dict[str, ParamSpec],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        params = params_by_id.get(result.params_id)
        delay = params.delay if params is not None else 7
        for segment_index, (start, end) in enumerate(
            contiguous_segments(result.label), start=1
        ):
            window_end = min(start + delay, end)
            timely_window = result.raw_pred[start : window_end + 1]
            segment_raw = result.raw_pred[start : end + 1]
            raw_offsets = np.where(segment_raw > 0)[0]
            if np.any(timely_window > 0):
                hit_type = "timely"
            elif raw_offsets.size > 0:
                hit_type = "late"
            else:
                hit_type = "no_pred"

            first_raw_offset = int(raw_offsets[0]) if raw_offsets.size > 0 else ""
            first_raw_index = start + first_raw_offset if raw_offsets.size > 0 else ""
            adjusted_positive = int(result.adjusted_pred[start : end + 1].sum())
            rows.append(
                {
                    "dataset": result.dataset,
                    "metric_name": result.metric_name,
                    "params_id": result.params_id,
                    "segment_index": segment_index,
                    "start_index": start,
                    "end_index": end,
                    "length": end - start + 1,
                    "start_timestamp": int(result.timestamp[start]),
                    "end_timestamp": int(result.timestamp[end]),
                    "delay": delay,
                    "hit_type": hit_type,
                    "first_raw_pred_offset": first_raw_offset,
                    "first_raw_pred_index": first_raw_index,
                    "raw_positive_in_segment": int(segment_raw.sum()),
                    "adjusted_positive_in_segment": adjusted_positive,
                    "adjusted_recall_in_segment": adjusted_positive / (end - start + 1),
                }
            )
    return rows


def plot_result(output_dir: Path, series: SeriesData, result: PredictionResult) -> None:
    plot_dir = output_dir / "plots" / result.dataset
    plot_dir.mkdir(parents=True, exist_ok=True)
    train_x = np.arange(len(series.train.value))
    test_x = np.arange(len(series.train.value), len(series.train.value) + len(result.value))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_x, series.train.value, color="#2f6fbb", linewidth=0.9, label="train")
    ax.plot(test_x, result.value, color="#606060", linewidth=0.9, label="test")
    ax.axvline(len(series.train.value), color="#999999", linewidth=0.8, linestyle="--")

    for start, end in contiguous_segments(result.label):
        ax.axvspan(test_x[start], test_x[end], color="#d62728", alpha=0.18)

    pred_idx = np.where(result.adjusted_pred > 0)[0]
    if pred_idx.size:
        ax.scatter(
            test_x[pred_idx],
            result.value[pred_idx],
            s=10,
            color="#ff8c00",
            label="FluxEV adjusted pred",
            zorder=3,
        )

    missing_idx = np.where(result.missing > 0)[0]
    if missing_idx.size:
        ax.scatter(
            test_x[missing_idx],
            result.value[missing_idx],
            s=8,
            color="#8a8a8a",
            label="filled missing",
            zorder=2,
        )

    ax.set_title(f"{result.dataset} / {result.metric_name} / {result.params_id}")
    ax.set_xlabel("completed point index")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{Path(result.metric_name).stem}.png", dpi=140)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    series_list: Sequence[SeriesData],
    results: Sequence[PredictionResult],
) -> List[Dict[str, object]]:
    series_by_key = {result_key(series): series for series in series_list}
    errors: List[Dict[str, object]] = []
    for result in results:
        try:
            plot_result(output_dir, series_by_key[result_key(result)], result)
        except Exception as exc:
            errors.append(
                {
                    "dataset": result.dataset,
                    "metric_name": result.metric_name,
                    "params_id": result.params_id,
                    "error": repr(exc),
                }
            )
    if errors:
        write_csv(output_dir / "plot_errors.csv", errors)
    return errors


def write_markdown_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "| 数据集 | 算法 | F1 | Precision | Recall |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['数据集']} | {row['算法']} | "
            f"{float(row['F1']):.4f} | {float(row['Precision']):.4f} | {float(row['Recall']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_rows(
    best_result_sets: Dict[str, Sequence[PredictionResult]]
) -> List[Dict[str, object]]:
    jmeter = [result for result in best_result_sets["jmeter"] if result.dataset == "jmeter"]
    prom_all = [result for result in best_result_sets["prometheus"] if result.dataset == "prometheus"]
    prom_valid = [result for result in prom_all if np.any(result.label == 1)]
    groups = [
        ("JMeter", jmeter),
        (f"Prometheus 有效 {len(prom_valid)} 指标", prom_valid),
        (f"Prometheus 全 {len(prom_all)} 指标", prom_all),
    ]
    final_rows: List[Dict[str, object]] = []
    for dataset_label, group_results in groups:
        metrics = pooled_metrics(group_results)
        final_rows.append(
            {
                "数据集": dataset_label,
                "算法": ALGORITHM_NAME,
                "F1": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
            }
        )
    return final_rows


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_summary: Dict[str, object],
    elapsed_seconds: float,
    best_params: Dict[str, str],
    selection_strategy: str,
    baseline_report: Optional[Path],
    plot_errors: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    metadata = {
        "algorithm": ALGORITHM_NAME,
        "run_dir": str(output_dir),
        "dataset_root": str(args.dataset_root),
        "mode": args.mode,
        "grid_preset": args.grid_preset,
        "selection_strategy": selection_strategy,
        "baseline_report": "" if baseline_report is None else str(baseline_report),
        "interval_seconds": args.interval_seconds,
        "elapsed_seconds": elapsed_seconds,
        "best_params": best_params,
        "dataset_summary": dataset_summary,
        "plot_error_count": len(plot_errors or []),
        "windows_environment_variables_modified": False,
        "gpu_used": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    fluxev_root = Path(__file__).resolve().parent.parent
    selection_strategy = args.selection_strategy
    if selection_strategy is None:
        selection_strategy = (
            "near_f1_max_recall" if args.grid_preset == "recall_fine" else "max_f1"
        )
    baseline_report = args.baseline_report or find_default_baseline_report(fluxev_root)
    if args.mode == "plots-only":
        if args.existing_output is None:
            raise ValueError("--existing-output is required for plots-only mode")
        output_dir = args.existing_output
    else:
        run_id = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.mode == "smoke" and args.timestamp is None:
            run_id = f"{run_id}_smoke"
        output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    print(f"[FluxEV] output_dir={output_dir}")
    print(f"[FluxEV] loading dataset from {args.dataset_root}")
    series_list = load_dataset(args.dataset_root, args.interval_seconds)
    dataset_summary = validate_dataset(series_list)
    print(f"[FluxEV] dataset_summary={json.dumps(dataset_summary, ensure_ascii=False)}")

    if args.mode == "plots-only":
        print("[FluxEV] plots-only reading best params", flush=True)
        best_ids = read_best_params(output_dir)
        print(f"[FluxEV] plots-only best_params={best_ids}", flush=True)
        print("[FluxEV] plots-only loading prediction CSV files", flush=True)
        best_predictions = load_existing_best_predictions(output_dir, series_list)
        print(f"[FluxEV] plots-only loaded predictions={len(best_predictions)}", flush=True)
        plot_errors: List[Dict[str, object]] = []
        if not args.no_plots:
            print("[FluxEV] plots-only writing plots", flush=True)
            plot_errors = write_plots(output_dir, series_list, best_predictions)
        elapsed_seconds = time.perf_counter() - start_time
        metadata_path = output_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["plots_only_elapsed_seconds"] = elapsed_seconds
            metadata["plot_error_count"] = len(plot_errors)
            metadata["windows_environment_variables_modified"] = False
            metadata["gpu_used"] = False
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            write_run_metadata(
                output_dir,
                args,
                dataset_summary,
                elapsed_seconds,
                best_ids,
                selection_strategy,
                baseline_report,
                plot_errors,
            )
        print(f"[FluxEV] plots-only wrote {len(best_predictions) - len(plot_errors)} plots")
        print(f"[FluxEV] plot_errors={len(plot_errors)}")
        print(f"[FluxEV] elapsed_seconds={elapsed_seconds:.2f}")
        return

    params_grid = build_param_grid(args.mode, args.grid_preset)
    params_by_id = {params.params_id: params for params in params_grid}
    print(f"[FluxEV] parameter_sets={len(params_grid)}")
    print(f"[FluxEV] grid_preset={args.grid_preset}")
    print(f"[FluxEV] selection_strategy={selection_strategy}")
    print(
        "[FluxEV] baseline_report="
        f"{baseline_report if baseline_report is not None else ''}"
    )

    grid_rows: List[Dict[str, object]] = []
    all_results_by_param: Dict[str, List[PredictionResult]] = {}
    for index, params in enumerate(params_grid, start=1):
        param_start = time.perf_counter()
        print(f"[FluxEV] running {index}/{len(params_grid)} {params.params_id}")
        results = run_param(series_list, params)
        all_results_by_param[params.params_id] = results
        rows = summarize_param(params, results)
        grid_rows.extend(rows)
        elapsed = time.perf_counter() - param_start
        print(f"[FluxEV] finished {params.params_id} in {elapsed:.2f}s")

    write_csv(output_dir / "grid_summary.csv", grid_rows)
    baseline_metrics = load_baseline_metrics(baseline_report)
    selection = select_best_params(grid_rows, selection_strategy, baseline_metrics)
    best_ids = selection_best_ids(selection)
    max_f1_ids = selection_max_f1_ids(selection)
    write_csv(output_dir / "best_params.csv", build_best_params_rows(selection))
    write_csv(
        output_dir / "recall_tradeoff_summary.csv",
        build_recall_tradeoff_rows(grid_rows, selection, baseline_metrics),
    )

    best_result_sets = {
        "jmeter": all_results_by_param[best_ids["jmeter"]],
        "prometheus": all_results_by_param[best_ids["prometheus"]],
    }
    final_rows = build_final_rows(best_result_sets)
    write_csv(output_dir / "final_report_table.csv", final_rows)
    write_markdown_table(output_dir / "final_report_table.md", final_rows)

    max_f1_result_sets = {
        "jmeter": all_results_by_param[max_f1_ids["jmeter"]],
        "prometheus": all_results_by_param[max_f1_ids["prometheus"]],
    }
    max_f1_rows = build_final_rows(max_f1_result_sets)
    write_csv(output_dir / "final_report_table_max_f1.csv", max_f1_rows)
    write_markdown_table(output_dir / "final_report_table_max_f1.md", max_f1_rows)

    best_predictions: List[PredictionResult] = []
    for result in best_result_sets["jmeter"]:
        if result.dataset == "jmeter":
            best_predictions.append(result)
    for result in best_result_sets["prometheus"]:
        if result.dataset == "prometheus":
            best_predictions.append(result)
    per_file_rows = [per_file_metrics(result) for result in best_predictions]
    write_csv(output_dir / "best_per_file_metrics.csv", per_file_rows)
    write_csv(
        output_dir / "segment_delay_analysis.csv",
        build_segment_delay_rows(best_predictions, params_by_id),
    )

    if not args.no_predictions:
        write_predictions(output_dir, best_predictions)
    plot_errors: List[Dict[str, object]] = []
    if not args.no_plots:
        plot_errors = write_plots(output_dir, series_list, best_predictions)

    elapsed_seconds = time.perf_counter() - start_time
    write_run_metadata(
        output_dir,
        args,
        dataset_summary,
        elapsed_seconds,
        best_ids,
        selection_strategy,
        baseline_report,
        plot_errors,
    )

    print("[FluxEV] final_report_table")
    for row in final_rows:
        print(
            f"  {row['数据集']} | {row['算法']} | "
            f"F1={float(row['F1']):.4f} | "
            f"Precision={float(row['Precision']):.4f} | "
            f"Recall={float(row['Recall']):.4f}"
        )
    print(f"[FluxEV] best_params={best_ids}")
    print(f"[FluxEV] max_f1_params={max_f1_ids}")
    print(f"[FluxEV] elapsed_seconds={elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
