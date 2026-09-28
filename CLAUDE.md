# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Research project comparing concept-drift detectors (ADWIN, KSWIN, DDM, Page-Hinkley from `river`) on streaming data, with the long-term goal of a SHAP-based drift detector (SHAP values per chunk, LASSO-driven window sizing). The SHAP part is not implemented yet — current code is the detector-benchmark baseline. Comments, prints and plot labels are in Polish; keep that convention.

## Environment and commands

- Windows, local virtualenv in `venv/`; dependencies pinned in `requirements.txt` (`../venv/Scripts/python -m pip install -r requirements.txt`). No tests or linter config.
- **All scripts use paths relative to `src/`** (`../data/...`), so run them from inside `src/`:

```
cd src
../venv/Scripts/python main.py                   # run all detectors on every dataset -> ../data/results/drift_detectors_results[_N].csv
../venv/Scripts/python stats.py --csv ../data/results/drift_detectors_results.csv   # summary table per detector -> detector_stats[_N].csv
../venv/Scripts/python results_visualization.py  # plots -> ../data/graphs/
../venv/Scripts/python drift_types_plot.py       # illustrative drift-type figures (standalone)
```

Output CSVs are never overwritten: `main.py` and `stats.py` append `_1`, `_2`, … to the filename if it exists, while `stats.py` and `results_visualization.py` read the un-suffixed file by default — pass the specific run you want.

## Pipeline architecture

1. **Datasets** — `data/datasets/*.arff` (gitignored). Ground truth is encoded **only in the filename**, parsed by `parse_filename` in `main.py`:
   `Name_f_F1_F2..._p_P1_P2..._w_W1_W2..._s_S_r_R.arff` — `f` drifting features, `p` drift points, `w` drift widths (one per drift point), `s` sample count, `r` seed. `p`/`w` may be absent (no-drift stream). Target column is `class`.
2. **`main.py` / `evaluate_stream`** — builds a fresh `river` pipeline per dataset (numeric passthrough + one-hot for strings → `GaussianNB`), warm-starts on the first 100 samples, then does test-then-train. The 0/1 classification error is fed to all four detectors; detection indices are recorded as `i + 100` (absolute stream position).
3. **Metrics** — `D1D2_metrics.py` holds static-style methods on class `D1D2` (called as `D1D2.tpr(...)`, no instance):
   - `get_window(tg, W)` → `[tg - W/2, tg + W/2 + 500]` (500 = detection-delay tolerance). All window-based metrics (TPR, FDR, FEDP, EDDR) go through it.
   - `D1` = mean distance from each detection to nearest true drift; `D2` = mean distance from each true drift to nearest detection.
   - `main.detection_rates` computes TPR/FDR with a separate width per drift and asserts drift windows don't overlap; `calculate_r` = `|true/detected - 1|`.
   - When a metric is undefined (no detections), `main.py` substitutes `samples_number` for R, D1, D2 as a penalty rather than leaving it null.
4. **Result CSV schema** — one row per dataset; per-detector columns follow `<DETECTOR>_<metric>` (`_detections`, `_detections_number`, `_false_discovery_rate`, `_true_positive_rate`, `_R`, `_D1`, `_D2`). Multi-valued cells (drift points, widths, detection lists) are joined with `"; "`; when read back, pandas yields a number for single values, a string for several, and NaN for empty, so parse them with `parse_multi_value` in `results_visualization.py`. `stats.py` and `results_visualization.py` rely on this naming — keep it consistent when adding detectors or metrics.
5. **`stats.py` summary** — one row per detector, every cell a formatted string ready for a paper: TPR/FDR as mean with a percentile-bootstrap 95% CI (`--n-boot`, `--seed`), counts of TPR=1/0 and FDR=0/1, N for FDR, R as mean with bootstrap 95% CI over all runs and over runs with ≥1 alarm, and D1/D2 as median [Q1; Q3] over all runs plus median [Q1; Q3] and P90 over runs with ≥1 alarm. Quantiles use `method="inverted_cdf"` (no interpolation). Subsets are chosen deliberately: no-drift datasets (empty `Drift_Point`) count only toward FDR; the "all runs" D1/D2/R stats include the `samples_number` penalty values, while the "with detection" stats exclude them; FDR uses only datasets with ≥1 alarm.
