"""Summarize SR-CNN saved scores on an optional Donut-valid file subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srcnn.competition_metric import evaluate_for_all_series  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SR-CNN saved scores.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--saved-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument(
        "--donut-summary",
        type=Path,
        default=None,
        help="Optional Donut summary.json. Uses its ok files as the fair subset.",
    )
    parser.add_argument(
        "--include-file-names",
        type=Path,
        default=None,
        help="Optional newline-separated basenames to include.",
    )
    parser.add_argument(
        "--align-to-donut-scores",
        action="store_true",
        help="Drop early SR-CNN points before Donut's first scored timestamp.",
    )
    return parser.parse_args()


def metric_dict(raw_metrics: Sequence[float]) -> Dict[str, float]:
    f1, precision, recall, tp, fp, tn, fn = raw_metrics
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def evaluate_at_threshold(
    saved_scores: Sequence[Sequence[object]],
    threshold: float,
    delay: int,
) -> Dict[str, float]:
    results = []
    for labels, scores, path, timestamps in saved_scores:
        preds = [1 if float(score) > threshold else 0 for score in scores]
        results.append([timestamps, labels, preds, path])
    return metric_dict(evaluate_for_all_series(results, delay, prt=False))


def best_threshold(
    saved_scores: Sequence[Sequence[object]],
    delay: int,
) -> Tuple[float, Dict[str, float]]:
    best = (0.0, None, None)
    for i in range(98):
        threshold = 0.01 + i * 0.01
        current = evaluate_at_threshold(saved_scores, threshold, delay)
        if current["f1"] > best[0]:
            best = (current["f1"], threshold, current)
    if best[1] is None:
        empty = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "TP": 0, "FP": 0, "TN": 0, "FN": 0}
        return 0.0, empty
    return float(best[1]), best[2]


def read_first_timestamp(path: Path) -> Optional[int]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return int(float(row["timestamp"]))
    return None


def resolve_score_path(donut_summary_path: Path, score_path: str) -> Path:
    path = Path(score_path)
    candidates = [
        path,
        Path.cwd() / path,
        donut_summary_path.parent.parent / path,
        donut_summary_path.parent / "scores" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_donut_subset(
    donut_summary_path: Path,
    align_to_donut_scores: bool,
) -> Tuple[Set[str], Dict[str, int]]:
    donut_summary_path = donut_summary_path.resolve()
    summary = json.loads(donut_summary_path.read_text(encoding="utf-8"))
    include_names = {
        Path(item["file"]).name
        for item in summary.get("files", [])
        if item.get("status") == "ok"
    }
    align_start: Dict[str, int] = {}
    if align_to_donut_scores:
        for item in summary.get("files", []):
            if item.get("status") != "ok" or not item.get("scores_path"):
                continue
            score_path = resolve_score_path(donut_summary_path, str(item["scores_path"]))
            if score_path.exists():
                first_timestamp = read_first_timestamp(score_path)
                if first_timestamp is not None:
                    align_start[Path(item["file"]).name] = first_timestamp
    return include_names, align_start


def load_include_names(path: Path) -> Set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def filter_saved_scores(
    saved_scores: Sequence[Sequence[object]],
    include_names: Iterable[str] | None,
    align_start: Dict[str, int],
) -> List[List[object]]:
    include_set = set(include_names) if include_names is not None else None
    filtered = []
    for labels, scores, path, timestamps in saved_scores:
        name = Path(str(path)).name
        if include_set is not None and name not in include_set:
            continue

        if name in align_start:
            start = align_start[name]
            rows = [
                (int(ts), int(label), float(score))
                for ts, label, score in zip(timestamps, labels, scores)
                if int(ts) >= start
            ]
            if not rows:
                continue
            aligned_timestamps, aligned_labels, aligned_scores = zip(*rows)
            filtered.append(
                [
                    list(aligned_labels),
                    list(aligned_scores),
                    path,
                    list(aligned_timestamps),
                ]
            )
        else:
            filtered.append([labels, scores, path, timestamps])
    return filtered


def main() -> None:
    args = parse_args()
    eval_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    saved_scores = json.loads(args.saved_scores.read_text(encoding="utf-8"))

    include_names = None
    align_start: Dict[str, int] = {}
    subset_source = "all_files"
    if args.donut_summary:
        include_names, align_start = load_donut_subset(
            args.donut_summary,
            args.align_to_donut_scores,
        )
        subset_source = str(args.donut_summary)
    elif args.include_file_names:
        include_names = load_include_names(args.include_file_names)
        subset_source = str(args.include_file_names)

    filtered = filter_saved_scores(saved_scores, include_names, align_start)
    initial_threshold = float(eval_summary.get("initial_threshold", 0.95))
    initial_metrics = evaluate_at_threshold(filtered, initial_threshold, args.delay)
    best, best_metrics = best_threshold(filtered, args.delay)

    output = {
        "source_summary": str(args.summary),
        "source_saved_scores": str(args.saved_scores),
        "subset_source": subset_source,
        "align_to_donut_scores": bool(args.align_to_donut_scores),
        "files_total_in_saved_scores": len(saved_scores),
        "files_used": len(filtered),
        "file_names_used": sorted(Path(str(item[2])).name for item in filtered),
        "delay": args.delay,
        "initial_threshold": initial_threshold,
        "initial_metrics": initial_metrics,
        "best_threshold": best,
        "best_metrics": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {args.output}")
    print("files_used:", len(filtered))
    print("best_metrics:", best_metrics)


if __name__ == "__main__":
    main()
