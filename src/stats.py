"""Compute mean and std of detector metrics and display as table.

Usage:
  python compute_detector_stats.py [--csv PATH] [--detectors LIST] [--metrics LIST]

Defaults:
  csv: ../results/drift_detectors_stats.csv
  detectors: ADWIN,KSWIN,DDM,PHT
  metrics (column suffixes after "<DETECTOR>_"): false_discovery_rate,true_positive_rate,R,D1,D2

Outputs a table with columns: detector, <metric>_mean, <metric>_std
"""
import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Compute mean and std of detector metrics and display as table")

    parser.add_argument("--csv", default="../data/results/drift_detectors_results_1.csv", help="Path to CSV file")
    parser.add_argument("--detectors", default="ADWIN,KSWIN,DDM,PHT", help="Comma-separated detector name prefixes")
    parser.add_argument(
        "--metrics",
        default="false_discovery_rate,true_positive_rate,R,D1,D2",
        help="Comma-separated metric column suffixes (after '<DETECTOR>_')",
    )
    parser.add_argument("--output", default="../data/results/detector_stats.csv", help="Path to save the output table (CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    rows = []
    missing = []
    for det in detectors:
        row = {"detector": det}
        for m in metrics:
            col = f"{det}_{m}"
            if col not in df.columns:
                missing.append(col)
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            row[f"{m}_mean"] = values.mean()
            row[f"{m}_std"] = values.std(ddof=0)
        rows.append(row)

    if missing:
        print(f"Columns not found in CSV: {', '.join(missing)}. Available columns: {', '.join(df.columns)}", file=sys.stderr)

    result = pd.DataFrame(rows)
    cols = ["detector"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    result = result.reindex(columns=cols)

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(result.to_string(index=False))


    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        original_output_path = output_path
        counter = 1
        # Szukamy dostępnej nazwy pliku
        while output_path.exists():
            output_path = original_output_path.parent / f"{original_output_path.stem}_{counter}{original_output_path.suffix}"
            counter += 1

        # Zapisujemy nowy plik (zawsze zapisujemy pełny DataFrame w nowym pliku)
        result.to_csv(output_path, index=False, float_format="%.6f")
        print(f"\nWszystkie wyniki zapisane poprawnie do: {output_path.as_posix()}")
    except Exception as e:
        print(f"Błąd przy zapisie do pliku {output_path.as_posix()}: {e}")


if __name__ == "__main__":
    main()