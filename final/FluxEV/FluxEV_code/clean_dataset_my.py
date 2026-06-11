import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


GROUPS = ("jmeter", "prometheus")
SUBDIRS = ("train_jmeter", "test_jmeter", "train_prometheus", "test_prometheus")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    fluxev_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Create a cleaned dataset_my copy without constant/binary KPI CSVs."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=fluxev_root / "dataset_my",
        help="Original dataset_my directory.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=fluxev_root / "dataset_my_clean_no_const_binary",
        help="Output directory for the cleaned dataset.",
    )
    parser.add_argument(
        "--max-unique-remove",
        type=int,
        default=2,
        help="Remove a paired KPI when train or test value unique count is <= this threshold.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target directory if it already exists.",
    )
    return parser.parse_args()


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


def value_unique_count(path: Path) -> Tuple[int, int, int]:
    values = set()
    rows = 0
    label_sum = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "value" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not have a value column")
        has_label = "label" in (reader.fieldnames or [])
        for row in reader:
            rows += 1
            raw_value = row.get("value", "")
            try:
                key = repr(float(raw_value))
            except ValueError:
                key = raw_value
            values.add(key)
            if has_label:
                try:
                    if float(row.get("label", "0")) > 0:
                        label_sum += 1
                except ValueError:
                    pass
    return len(values), rows, label_sum


def clean_target(target: Path, overwrite: bool) -> None:
    if not target.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Target already exists: {target}. Use --overwrite to replace it.")

    resolved = target.resolve()
    if resolved.name != "dataset_my_clean_no_const_binary":
        raise ValueError(f"Refusing to overwrite unexpected target: {resolved}")
    shutil.rmtree(resolved)


def copy_csv(src: Path, source_subdir: Path, target_subdir: Path) -> None:
    relative = src.relative_to(source_subdir)
    dst = target_subdir / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_manifest(source: Path, target: Path, max_unique_remove: int) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []

    for group in GROUPS:
        train_subdir = source / f"train_{group}"
        test_subdir = source / f"test_{group}"
        train_files = collect_csv_by_basename(train_subdir)
        test_files = collect_csv_by_basename(test_subdir)

        missing_train = sorted(set(test_files) - set(train_files))
        missing_test = sorted(set(train_files) - set(test_files))
        if missing_train or missing_test:
            raise ValueError(
                f"{group} train/test mismatch. "
                f"missing_train={missing_train}, missing_test={missing_test}"
            )

        for basename in sorted(test_files):
            train_unique, train_rows, _ = value_unique_count(train_files[basename])
            test_unique, test_rows, test_label_sum = value_unique_count(test_files[basename])
            remove = train_unique <= max_unique_remove or test_unique <= max_unique_remove
            reason = ""
            if remove:
                reasons = []
                if train_unique <= max_unique_remove:
                    reasons.append(f"train_unique<={max_unique_remove}")
                if test_unique <= max_unique_remove:
                    reasons.append(f"test_unique<={max_unique_remove}")
                reason = ";".join(reasons)
            else:
                copy_csv(train_files[basename], train_subdir, target / f"train_{group}")
                copy_csv(test_files[basename], test_subdir, target / f"test_{group}")

            manifest.append(
                {
                    "group": group,
                    "basename": basename,
                    "kept": int(not remove),
                    "removed": int(remove),
                    "reason": reason,
                    "train_unique": train_unique,
                    "test_unique": test_unique,
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "test_label_sum": test_label_sum,
                    "test_label_ratio": f"{(test_label_sum / test_rows) if test_rows else 0:.6f}",
                    "train_path": str(train_files[basename].relative_to(source)),
                    "test_path": str(test_files[basename].relative_to(source)),
                }
            )

    return manifest


def write_manifest(target: Path, manifest: List[Dict[str, object]]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "clean_manifest.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)


def write_readme(target: Path, source: Path, manifest: List[Dict[str, object]], threshold: int) -> None:
    def counts(group: str) -> Tuple[int, int, int]:
        rows = [row for row in manifest if row["group"] == group]
        kept = sum(int(row["kept"]) for row in rows)
        removed = sum(int(row["removed"]) for row in rows)
        return len(rows), kept, removed

    j_total, j_kept, j_removed = counts("jmeter")
    p_total, p_kept, p_removed = counts("prometheus")
    total = len(manifest)
    kept = sum(int(row["kept"]) for row in manifest)
    removed = sum(int(row["removed"]) for row in manifest)

    readme = f"""# dataset_my_clean_no_const_binary

This directory is a cleaned copy of:

`{source}`

Cleaning rule:

- Treat each train/test KPI pair as one unit.
- Remove the pair when either train or test `value` unique count is `<= {threshold}`.
- This removes constant and binary KPI curves.
- Do not remove JMeter curves only because their anomaly-label ratio is high.
- Do not copy `*_128_train.json`; those files are windowized auxiliary data and would no longer match the cleaned CSV set.

Summary:

| Group | Original CSV pairs | Kept | Removed |
| --- | ---: | ---: | ---: |
| JMeter | {j_total} | {j_kept} | {j_removed} |
| Prometheus | {p_total} | {p_kept} | {p_removed} |
| Total | {total} | {kept} | {removed} |

Details are in `clean_manifest.csv`.
"""
    (target / "README_cleaning.md").write_text(readme, encoding="utf-8")


def validate_output(target: Path, manifest: List[Dict[str, object]], threshold: int) -> None:
    for subdir in SUBDIRS:
        (target / subdir).mkdir(parents=True, exist_ok=True)

    for group in GROUPS:
        train_files = collect_csv_by_basename(target / f"train_{group}")
        test_files = collect_csv_by_basename(target / f"test_{group}")
        if set(train_files) != set(test_files):
            raise ValueError(f"Cleaned {group} train/test basenames do not match")

        for basename in sorted(train_files):
            train_unique, _, _ = value_unique_count(train_files[basename])
            test_unique, _, _ = value_unique_count(test_files[basename])
            if train_unique <= threshold or test_unique <= threshold:
                raise ValueError(
                    f"Kept low-information file {basename}: "
                    f"train_unique={train_unique}, test_unique={test_unique}"
                )

    kept_manifest = sum(int(row["kept"]) for row in manifest)
    actual_csvs = sum(1 for _ in target.rglob("*.csv"))
    # Two CSVs per kept KPI pair plus the manifest itself.
    expected_csvs = kept_manifest * 2 + 1
    if actual_csvs != expected_csvs:
        raise ValueError(f"Expected {expected_csvs} CSV files, found {actual_csvs}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    clean_target(target, args.overwrite)
    target.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(source, target, args.max_unique_remove)
    write_manifest(target, manifest)
    write_readme(target, source, manifest, args.max_unique_remove)
    validate_output(target, manifest, args.max_unique_remove)

    kept = sum(int(row["kept"]) for row in manifest)
    removed = sum(int(row["removed"]) for row in manifest)
    print(f"Cleaned dataset written to: {target}")
    print(f"Kept KPI pairs: {kept}")
    print(f"Removed KPI pairs: {removed}")


if __name__ == "__main__":
    main()
