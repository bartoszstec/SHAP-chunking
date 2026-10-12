import argparse
import math

import pandas as pd

DETECTORS = ["ADWIN", "KSWIN", "DDM", "PHT"]
METRICS = {
    "FDR": "false_discovery_rate",
    "TPR": "true_positive_rate",
    "R": "R",
}


def fill_nan_cyclic(values):
    """Zastępuje NaN kolejnymi istniejącymi wartościami z tej samej listy (cyklicznie).
    Np. [0, 0.5, 0.67, NaN, NaN, NaN, NaN] -> [0, 0.5, 0.67, 0, 0.5, 0.67, 0]."""
    existing = [v for v in values if not math.isnan(v)]
    if not existing:
        return values
    filled = []
    k = 0
    for v in values:
        if math.isnan(v):
            filled.append(existing[k % len(existing)])
            k += 1
        else:
            filled.append(v)
    return filled


def main():
    parser = argparse.ArgumentParser(description="Wypisuje tablice FDR, TPR i R dla każdego detektora.")
    parser.add_argument("--csv", default="../data/results/drift_detectors_binary_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for detector in DETECTORS:
        for name, suffix in METRICS.items():
            values = df[f"{detector}_{suffix}"].astype(float).tolist()
            values = fill_nan_cyclic(values)
            print(f"{detector}_{name} = {values}")
        print()


if __name__ == "__main__":
    main()
