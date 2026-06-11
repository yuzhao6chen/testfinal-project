import csv
import shutil
from pathlib import Path


def collect_csv(root: Path):
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def collect_csv_by_name(root: Path):
    mapping = {}
    for path in collect_csv(root):
        if path.name in mapping:
            raise ValueError(f"Duplicate CSV basename under {root}: {path.name}")
        mapping[path.name] = path
    return mapping


def label_stats(path: Path):
    rows = 0
    anomaly_points = 0
    anomaly_segments = 0
    prev = 0
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "label" not in (reader.fieldnames or []):
            raise ValueError(f"Missing label column: {path}")
        for row in reader:
            rows += 1
            label = int(float(row["label"]))
            if label == 1:
                anomaly_points += 1
                if prev == 0:
                    anomaly_segments += 1
            prev = label
    return rows, anomaly_points, anomaly_segments


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_tree_csv(source_root: Path, target_root: Path):
    copied = []
    for source in collect_csv(source_root):
        rel = source.relative_to(source_root)
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append((source, target, rel))
    return copied


def main():
    fluxev_root = Path(__file__).resolve().parent.parent
    source_root = fluxev_root / "dataset_my_clean"
    target_root = fluxev_root / "dataset_my_final"
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not target_root.exists():
        raise FileNotFoundError(target_root)

    train_source = source_root / "train_jmeter"
    test_source = source_root / "test_jmeter"
    train_target = target_root / "train_jmeter"
    test_target = target_root / "test_jmeter"
    train_copied = copy_tree_csv(train_source, train_target)
    test_copied = copy_tree_csv(test_source, test_target)

    train_by_name = collect_csv_by_name(train_target)
    manifest_rows = []
    for _, test_path, test_rel in test_copied:
        if test_path.name not in train_by_name:
            raise FileNotFoundError(f"Missing copied train CSV for {test_path.name}")
        train_rel = train_by_name[test_path.name].relative_to(train_target)
        rows, anomaly_points, anomaly_segments = label_stats(test_path)
        manifest_rows.append(
            {
                "metric_name": test_path.name,
                "train_relative_path": str(train_rel).replace("\\", "/"),
                "test_relative_path": str(test_rel).replace("\\", "/"),
                "test_rows": rows,
                "anomaly_points": anomaly_points,
                "anomaly_segments": anomaly_segments,
            }
        )
    write_csv(target_root / "manifest_jmeter_7.csv", manifest_rows)

    prom_train_source = source_root / "train_prometheus"
    prom_test_source = source_root / "test_prometheus"
    prom_train_by_name = collect_csv_by_name(prom_train_source)
    prom_test_by_name = collect_csv_by_name(prom_test_source)
    prom_manifest_rows = []
    excluded_rows = []
    for name, test_path in sorted(prom_test_by_name.items()):
        rows, anomaly_points, anomaly_segments = label_stats(test_path)
        if anomaly_points <= 0:
            excluded_rows.append(
                {
                    "metric_name": name,
                    "reason": "test_has_no_anomaly_label",
                    "test_rows": rows,
                    "anomaly_points": anomaly_points,
                    "anomaly_segments": anomaly_segments,
                }
            )
            continue
        if name not in prom_train_by_name:
            raise FileNotFoundError(f"Missing Prometheus train CSV for {name}")
        train_rel = prom_train_by_name[name].relative_to(prom_train_source)
        prom_manifest_rows.append(
            {
                "metric_name": name,
                "train_relative_path": str(train_rel).replace("\\", "/"),
                "test_relative_path": name,
                "test_rows": rows,
                "anomaly_points": anomaly_points,
                "anomaly_segments": anomaly_segments,
            }
        )
    write_csv(target_root / "manifest_effective_27.csv", prom_manifest_rows)
    write_csv(target_root / "excluded_no_anomaly_6.csv", excluded_rows)

    readme = "\n".join(
        [
            "# dataset_my_final",
            "",
            "This dataset is derived from dataset_my_clean.",
            "",
            "- Prometheus keeps only the 27 train/test pairs whose test CSV has at least one anomaly label.",
            "- JMeter is copied unchanged from dataset_my_clean: 7 train CSVs and 7 test CSVs.",
            "- Prometheus train CSVs keep their original nested relative paths.",
            "- Prometheus test CSVs are stored flat under test_prometheus.",
            "- JMeter train/test directories preserve the original source directory structure.",
            "",
            "Manifest files:",
            "",
            "- manifest_effective_27.csv: kept Prometheus files.",
            "- excluded_no_anomaly_6.csv: excluded all-normal Prometheus test CSVs.",
            "- manifest_jmeter_7.csv: copied JMeter files.",
            "",
        ]
    )
    (target_root / "README.md").write_text(readme, encoding="utf-8")

    print(f"target={target_root}")
    print(f"train_jmeter_copied={len(train_copied)}")
    print(f"test_jmeter_copied={len(test_copied)}")
    print(f"manifest_jmeter_rows={len(manifest_rows)}")
    print(f"manifest_prometheus_rows={len(prom_manifest_rows)}")
    print(f"excluded_prometheus_rows={len(excluded_rows)}")


if __name__ == "__main__":
    main()
