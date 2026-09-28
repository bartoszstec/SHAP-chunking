"""Compute summary statistics of detector metrics and display them as a table.

Usage:
  python stats.py [--csv PATH] [--detectors LIST] [--output PATH] [--n-boot N] [--seed S]

Defaults:
  csv: ../data/results/drift_detectors_results.csv
  detectors: ADWIN,KSWIN,DDM,PHT
  output: ../data/results/detector_stats.csv

One row per detector, cells are formatted strings:
  TPR                 mean [95% CI]           (bootstrap, datasets with a true drift)
  TPR=1; TPR=0        counts of datasets
  FDR                 mean [95% CI]           (bootstrap, datasets with >= 1 alarm)
  FDR=0; FDR=1        counts of datasets
  N_FDR               number of datasets with >= 1 alarm
  R_all               mean [95% CI]           (bootstrap, datasets with a true drift, penalty values included)
  R_det               mean [95% CI]           (bootstrap, datasets with a true drift and >= 1 alarm)
  D1/D2_all           median [Q1; Q3]         (datasets with a true drift, penalty values included)
  D1/D2_det           median [Q1; Q3]         (datasets with a true drift and >= 1 alarm)
  D1/D2_det_P90       90th percentile         (as above)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

QUANTILE_METHOD = "inverted_cdf"   # typ 1 wg Hyndmana i Fana (1996), bez interpolacji



def bootstrap_mean_ci(values, n_boot, rng, level=0.95):
    """Średnia i przedział ufności bootstrap (metoda percentylowa)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return values.mean(), low, high


def fmt_mean_ci(values, n_boot, rng):
    result = bootstrap_mean_ci(values, n_boot, rng)
    if result is None:
        return "n/a"
    mean, low, high = result
    return f"{mean:.2f} [{low:.2f}; {high:.2f}]"


def fmt_median_iqr(values, decimals):
    if len(values) == 0:
        return "n/a"
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75], method=QUANTILE_METHOD)
    return f"{med:.{decimals}f} [{q1:.{decimals}f}; {q3:.{decimals}f}]"


def fmt_p90(values, decimals):
    if len(values) == 0:
        return "n/a"
    return f"{np.quantile(values, 0.9, method=QUANTILE_METHOD):.{decimals}f}"


def detector_stats(df, det, n_boot, rng):
    has_drift = df["Drift_Point"].notna()
    has_alarm = pd.to_numeric(df[f"{det}_detections_number"], errors="coerce") > 0

    # TPR - tylko zbiory z prawdziwym dryftem (dla zbiorów bez dryftu TPR jest niezdefiniowane)
    tpr = pd.to_numeric(df.loc[has_drift, f"{det}_true_positive_rate"], errors="coerce").dropna()
    # FDR - wszystkie zbiory z co najmniej jednym alarmem (także bez dryftu: tam każdy alarm jest fałszywy)
    fdr = pd.to_numeric(df.loc[has_alarm, f"{det}_false_discovery_rate"], errors="coerce").dropna()

    row = {
        "detector": det,
        "TPR mean [95% CI]": fmt_mean_ci(tpr, n_boot, rng),
        "TPR=1; TPR=0": f"{int((tpr == 1).sum())}; {int((tpr == 0).sum())}",
        "FDR mean [95% CI]": fmt_mean_ci(fdr, n_boot, rng),
        "FDR=0; FDR=1": f"{int((fdr == 0).sum())}; {int((fdr == 1).sum())}",
        "N_FDR": len(fdr),
    }

    # R - tylko zbiory z dryftem; "all" zawiera wartości kary (samples_number) przy braku detekcji
    r = pd.to_numeric(df[f"{det}_R"], errors="coerce")
    row["R_all mean [95% CI]"] = fmt_mean_ci(r[has_drift].dropna(), n_boot, rng)
    row["R_det mean [95% CI]"] = fmt_mean_ci(r[has_drift & has_alarm].dropna(), n_boot, rng)

    # D1, D2 - tylko zbiory z dryftem; "all" zawiera wartości kary (samples_number) przy braku detekcji
    decimals = {"D1": 0, "D2": 0}
    for m, dec in decimals.items():
        col = pd.to_numeric(df[f"{det}_{m}"], errors="coerce")
        all_runs = col[has_drift].dropna()
        det_runs = col[has_drift & has_alarm].dropna()
        row[f"{m}_all median [Q1; Q3]"] = fmt_median_iqr(all_runs, dec)
        row[f"{m}_det median [Q1; Q3]"] = fmt_median_iqr(det_runs, dec)
        row[f"{m}_det P90"] = fmt_p90(det_runs, dec)

    return row


def main():
    parser = argparse.ArgumentParser(description="Compute summary statistics of detector metrics and display them as a table")

    parser.add_argument("--csv", default="../data/results/drift_detectors_results.csv", help="Path to CSV file")
    parser.add_argument("--detectors", default="ADWIN,KSWIN,DDM,PHT", help="Comma-separated detector name prefixes")
    parser.add_argument("--output", default="../data/results/detector_stats.csv", help="Path to save the output table (CSV)")
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap resamples for 95%% CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    required = ["_detections_number", "_true_positive_rate", "_false_discovery_rate", "_D1", "_D2", "_R"]
    missing = [f"{det}{suffix}" for det in detectors for suffix in required if f"{det}{suffix}" not in df.columns]
    if missing:
        print(f"Columns not found in CSV: {', '.join(missing)}. Available columns: {', '.join(df.columns)}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    result = pd.DataFrame([detector_stats(df, det, args.n_boot, rng) for det in detectors])

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
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
        result.to_csv(output_path, index=False)
        print(f"\nWszystkie wyniki zapisane poprawnie do: {output_path.as_posix()}")
    except Exception as e:
        print(f"Błąd przy zapisie do pliku {output_path.as_posix()}: {e}")


if __name__ == "__main__":
    main()
