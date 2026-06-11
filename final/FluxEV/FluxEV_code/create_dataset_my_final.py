import csv
import shutil
from pathlib import Path


def collect_csv_by_name(root: Path):
    mapping = {}
    for path in sorted(root.rglob("*.csv")):
        if path.name in mapping:
            raise ValueError(f"Duplicate CSV basename under {root}: {path.name}")
        mapping[path.name] = path
    return mapping


def collect_csv(root: Path):
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def copy_tree_csv(source_root: Path, target_root: Path):
    copied = []
    for source in collect_csv(source_root):
        rel = source.relative_to(source_root)
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append((source, target, rel))
    return copied


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


def main():
    fluxev_root = Path(__file__).resolve().parent.parent
    source_root = fluxev_root / "dataset_my_clean"
    target_root = fluxev_root / "dataset_my_final"

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if target_root.exists() and any(target_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {target_root}")

    prom_train_source = source_root / "train_prometheus"
    prom_test_source = source_root / "test_prometheus"
    prom_train_target = target_root / "train_prometheus"
    prom_test_target = target_root / "test_prometheus"

    train_by_name = collect_csv_by_name(prom_train_source)
    test_by_name = collect_csv_by_name(prom_test_source)

    prom_manifest_rows = []
    excluded_rows = []
    for name, test_path in sorted(test_by_name.items()):
        if name not in train_by_name:
            raise FileNotFoundError(f"Missing train CSV for {name}")
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

        train_path = train_by_name[name]
        train_rel = train_path.relative_to(prom_train_source)
        test_rel = Path(name)
        copied_train = prom_train_target / train_rel
        copied_test = prom_test_target / test_rel
        copied_train.parent.mkdir(parents=True, exist_ok=True)
        copied_test.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_path, copied_train)
        shutil.copy2(test_path, copied_test)
        prom_manifest_rows.append(
            {
                "metric_name": name,
                "train_relative_path": str(train_rel).replace("\\", "/"),
                "test_relative_path": str(test_rel).replace("\\", "/"),
                "test_rows": rows,
                "anomaly_points": anomaly_points,
                "anomaly_segments": anomaly_segments,
            }
        )

    jmeter_train_target = target_root / "train_jmeter"
    jmeter_test_target = target_root / "test_jmeter"
    copy_tree_csv(source_root / "train_jmeter", jmeter_train_target)
    jmeter_test_copied = copy_tree_csv(source_root / "test_jmeter", jmeter_test_target)
    jmeter_train_by_name = collect_csv_by_name(jmeter_train_target)
    jmeter_manifest_rows = []
    for _, copied_test, test_rel in jmeter_test_copied:
        if copied_test.name not in jmeter_train_by_name:
            raise FileNotFoundError(f"Missing JMeter train CSV for {copied_test.name}")
        train_rel = jmeter_train_by_name[copied_test.name].relative_to(jmeter_train_target)
        rows, anomaly_points, anomaly_segments = label_stats(copied_test)
        jmeter_manifest_rows.append(
            {
                "metric_name": copied_test.name,
                "train_relative_path": str(train_rel).replace("\\", "/"),
                "test_relative_path": str(test_rel).replace("\\", "/"),
                "test_rows": rows,
                "anomaly_points": anomaly_points,
                "anomaly_segments": anomaly_segments,
            }
        )

    write_csv(target_root / "manifest_effective_27.csv", prom_manifest_rows)
    write_csv(target_root / "excluded_no_anomaly_6.csv", excluded_rows)
    write_csv(target_root / "manifest_jmeter_7.csv", jmeter_manifest_rows)
    (target_root / "README.md").write_text(
        "\n".join(
            [
                "# dataset_my_final",
                "",
                "This dataset is derived from dataset_my_clean.",
                "",
                f"- Kept Prometheus train/test pairs with anomaly labels: {len(prom_manifest_rows)}",
                f"- Excluded all-normal Prometheus test CSVs: {len(excluded_rows)}",
                f"- Copied JMeter train/test pairs unchanged: {len(jmeter_manifest_rows)}",
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
        ),
        encoding="utf-8",
    )

    print(f"created={target_root}")
    print(f"kept_prometheus={len(prom_manifest_rows)}")
    print(f"excluded={len(excluded_rows)}")
    print(f"copied_jmeter={len(jmeter_manifest_rows)}")


if __name__ == "__main__":
    main()
