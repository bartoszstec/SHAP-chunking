from pathlib import Path
import pandas as pd
from scipy.io import arff
from river import stream, forest, metrics
from river import drift
import os
from D1D2_metrics import D1D2
from datetime import datetime

# Auxiliary functions
# R - adjusted ratio of the number of true drifts to the number of detections
def calculate_r(true_drifts_number, detected_drifts_number):
    if detected_drifts_number == 0:
        return None
    return abs((abs(true_drifts_number) / abs(detected_drifts_number)) - 1)

def evaluate_stream(model, dataset_path):
    # Detectors definition
    d_adwin = drift.ADWIN()
    # ADWIN Default parameters values: delta → 0.002, clock → 32, max_buckets → 5, min_window_length → 5, grace_period → 20
    d_kswin = drift.KSWIN(seed=42)
    # KSWIN Default parameters values: alpha → 0.005, window_size → 100, stat_size → 30, seed → None, window → None
    d_ddm = drift.binary.DDM()
    # DDM Default parameters values: warm_start → 30, warning_threshold → 2.0, drift_threshold → 3.0
    d_pht = drift.PageHinkley()
    # PHT Default parameters values: min_instances → 30, delta → 0.005, threshold → 50, alpha → 0.9999, mode → 'both'

    metric = metrics.Accuracy()  # wyświetlenie metryk

    # ----------
    # WARM START
    # ----------
    sample_range = 100
    dataset_init = stream.iter_arff(dataset_path, target='class')
    initial_data = [next(dataset_init) for _ in range(sample_range)]

    print(f"\nRozpoczynam rozgrzewanie modelu...")
    for i, (x, y) in enumerate(initial_data):
        # Skip first record if None
        if y is None:
            # print(f"Uwaga: Rekord {i} ma pustą etykietę (None)!")
            continue
        # Repair types
        y = y.decode('utf-8') if isinstance(y, bytes) else y

        try:
            model.learn_one(x, y)
        except TypeError as e:
            print(f"Błąd w rekordzie {i} przy wartości y={y} (typ: {type(y)}): {e}")

    print("Model rozgrzany.")

    # -----------------
    # PROPER DATA STREAM
    # -----------------
    drifts_found = {"ADWIN": [], "KSWIN": [], "DDM": [], "PHT": []}
    for i, (x, y) in enumerate(dataset_init):
        # Repair types
        y = y.decode('utf-8') if isinstance(y, bytes) else y

        # Prediction
        y_pred = model.predict_one(x)

        # Actualization
        if y_pred is not None:
            metric.update(y, y_pred)

            # Classification error
            error = 0 if y_pred == y else 1

            # Class probability
            #proba = model.predict_proba_one(x)
            #true_class_proba = proba.get(y, 0.0)

            # Drift detectors actualization
            d_adwin.update(error)                           # ADWIN
            d_kswin.update(error)                           # KSWIN
            #d_kswin.update(float(true_class_proba))       # KSWIN - alternative approach using class probability
            d_ddm.update(True if error == 1 else False)     # DDM
            d_pht.update(error)                             # PHT
            # Zamiast wrzucać do ADWIN-a informację o błędzie klasyfikacji (0 lub 1)
            # wrzucić wartości SHAP dla konkretnej cechy lub wektor ważności cech


            # Check if drift detected
            if d_adwin.drift_detected:
                # print(f"ADWIN wykrył dryft w rekordzie {i + sample_range}")
                drifts_found["ADWIN"].append(i + sample_range)
            if d_kswin.drift_detected:
                # print(f"KSWIN wykrył dryft w rekordzie {i + sample_range}")
                drifts_found["KSWIN"].append(i + sample_range)
            if d_ddm.drift_detected:
                # print(f"DDM wykrył dryft w rekordzie {i + sample_range}")
                drifts_found["DDM"].append(i + sample_range)
            if d_pht.drift_detected:
                # print(f"PHT wykrył dryft w rekordzie {i + sample_range}")
                drifts_found["PHT"].append(i + sample_range)


        # Online learning
        model.learn_one(x, y)

        # Metrics by 1000 records
        if i % 1000 == 0 and i > 0:
            print(f"Rekord: {i} | Aktualne Accuracy: {metric.get():.4f}")

    print(f"Zakończenie strumienia dla zbioru: {dataset_path}")

    # Extracting information from a file name
    # Formula: DatasetName_f_F1_F2_p_P_w_W_s_S_r_R
    # -------> F1, F2 - features used for drift, P - point of drift, W - width of drift, S - number of samples, R - random seed
    filename = dataset_path.split('/')[-1]
    parts = filename.split('_')
    point_drift = int(parts[parts.index('p')+1])
    width_drift = int(parts[parts.index('w')+1])
    samples_number = int(parts[parts.index('s') + 1])
    dataset_name = filename

    # -------Prepare values used for next calculations-------

    # Variables used for later calculations
    true_drifts = [point_drift]     # list of all true drifts (only one in this case)
    true_drifts_number = len(true_drifts)  # number of true drifts (only one in this case)

    # Sample points where drift was detected by each detector
    adwin_drifts = drifts_found.get("ADWIN", [])
    kswin_drifts = drifts_found.get("KSWIN", [])
    ddm_drifts = drifts_found.get("DDM", [])
    pht_drifts = drifts_found.get("PHT", [])

    # -------Detection statistics-------

    # DETECTIONS - all samples where drift was detected by each detector
    adwin_detections = "; ".join(map(str, adwin_drifts)) if adwin_drifts else ""
    kswin_detections = "; ".join(map(str, kswin_drifts)) if kswin_drifts else ""
    ddm_detections = "; ".join(map(str, ddm_drifts)) if ddm_drifts else ""
    pht_detections = "; ".join(map(str, pht_drifts)) if pht_drifts else ""

    # DETECTIONS NUMBER - number of all detections by each detector
    adwin_detections_number = len(adwin_drifts)
    kswin_detections_number = len(kswin_drifts)
    ddm_detections_number = len(ddm_drifts)
    pht_detections_number = len(pht_drifts)

    # FDR
    fdr_adwin = D1D2.fdr(true_drifts, adwin_drifts, width_drift)
    fdr_kswin = D1D2.fdr(true_drifts, kswin_drifts, width_drift)
    fdr_ddm = D1D2.fdr(true_drifts, ddm_drifts, width_drift)
    fdr_pht = D1D2.fdr(true_drifts, pht_drifts, width_drift)

    # TPR
    tpr_adwin = D1D2.tpr(true_drifts, adwin_drifts, width_drift)
    tpr_kswin = D1D2.tpr(true_drifts, kswin_drifts, width_drift)
    tpr_ddm = D1D2.tpr(true_drifts, ddm_drifts, width_drift)
    tpr_pht = D1D2.tpr(true_drifts, pht_drifts, width_drift)

    # R - adjusted ratio of the number of true drifts to the number of detections
    r_adwin = calculate_r(true_drifts_number, adwin_detections_number)
    r_kswin = calculate_r(true_drifts_number, kswin_detections_number)
    r_ddm = calculate_r(true_drifts_number, ddm_detections_number)
    r_pht = calculate_r(true_drifts_number, pht_detections_number)

    # D1 and D2 metrics calculations
    d1_adwin = D1D2.D1(true_drifts, adwin_drifts)
    d2_adwin = D1D2.D2(true_drifts, adwin_drifts)

    d1_kswin = D1D2.D1(true_drifts, kswin_drifts)
    d2_kswin = D1D2.D2(true_drifts, kswin_drifts)

    d1_ddm = D1D2.D1(true_drifts, ddm_drifts)
    d2_ddm = D1D2.D2(true_drifts, ddm_drifts)

    d1_pht = D1D2.D1(true_drifts, pht_drifts)
    d2_pht = D1D2.D2(true_drifts, pht_drifts)

    # Returns a dictionary with results for this dataset
    return {
        'Dataset': dataset_name,
        'Drift_Point': point_drift,
        'Width_Drift': width_drift,
        'Samples_Number': samples_number,

        'ADWIN_detections': adwin_detections,
        'KSWIN_detections': kswin_detections,
        'DDM_detections': ddm_detections,
        'PHT_detections': pht_detections,

        'ADWIN_detections_number': adwin_detections_number,
        'KSWIN_detections_number': kswin_detections_number,
        'DDM_detections_number': ddm_detections_number,
        'PHT_detections_number': pht_detections_number,

        'ADWIN_false_discovery_rate': round(fdr_adwin, 2) if fdr_adwin is not None else None,
        'KSWIN_false_discovery_rate': round(fdr_kswin, 2) if fdr_kswin is not None else None,
        'DDM_false_discovery_rate': round(fdr_ddm, 2) if fdr_ddm is not None else None,
        'PHT_false_discovery_rate': round(fdr_pht, 2) if fdr_pht is not None else None,

        'ADWIN_true_positive_rate': round(tpr_adwin, 2) if tpr_adwin is not None else None,
        'KSWIN_true_positive_rate': round(tpr_kswin, 2) if tpr_kswin is not None else None,
        'DDM_true_positive_rate': round(tpr_ddm, 2) if tpr_ddm is not None else None,
        'PHT_true_positive_rate': round(tpr_pht, 2) if tpr_pht is not None else None,

        'ADWIN_R': round(r_adwin, 2) if r_adwin is not None else samples_number,
        'KSWIN_R': round(r_kswin, 2) if r_kswin is not None else samples_number,
        'DDM_R': round(r_ddm, 2) if r_ddm is not None else samples_number,
        'PHT_R': round(r_pht, 2) if r_pht is not None else samples_number,

        'ADWIN_D1': round(d1_adwin) if d1_adwin is not None else samples_number,
        'ADWIN_D2': round(d2_adwin) if d2_adwin is not None else samples_number,
        'KSWIN_D1': round(d1_kswin) if d1_kswin is not None else samples_number,
        'KSWIN_D2': round(d2_kswin) if d2_kswin is not None else samples_number,
        'DDM_D1': round(d1_ddm) if d1_ddm is not None else samples_number,
        'DDM_D2': round(d2_ddm) if d2_ddm is not None else samples_number,
        'PHT_D1': round(d1_pht) if d1_pht is not None else samples_number,
        'PHT_D2': round(d2_pht) if d2_pht is not None else samples_number,

        'Ending_Accuracy': metric.get()
    }

def save_final_results(all_results_list):
    csv_path = Path("../data/results/drift_detection_results.csv")

    # Tworzymy DataFrame ze wszystkich wyników naraz
    df_results = pd.DataFrame(all_results_list)

    # Lista kolumn, które powinny być liczbami całkowitymi
    int_columns = [
        'Drift_Point', 'Width_Drift', 'Samples_Number',
        'ADWIN_all_detections', 'KSWIN_all_detections',
        'DDM_all_detections', 'PHT_all_detections',
        'ADWIN_D1', 'ADWIN_D2', 'KSWIN_D1', 'KSWIN_D2',
        'DDM_D1', 'DDM_D2', 'PHT_D1', 'PHT_D2'
    ]

    # Wymuszamy typ Int64 (przez duże I) - on obsługuje <null> i nie robi floatów
    for col in int_columns:
        if col in df_results.columns:
            df_results[col] = df_results[col].astype('Int64')

    # Zapis do pliku - jeśli plik istnieje, dopisujemy numer do nazwy (drift_detection_results_1.csv, _2, ...)
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        target_path = csv_path
        counter = 1
        # Szukamy dostępnej nazwy pliku
        while target_path.exists():
            target_path = csv_path.parent / f"{csv_path.stem}_{counter}{csv_path.suffix}"
            counter += 1

        # Zapisujemy nowy plik (zawsze zapisujemy pełny DataFrame w nowym pliku)
        df_results.to_csv(target_path, index=False)
        print(f"\nWszystkie wyniki zapisane poprawnie do: {target_path.as_posix()}")
    except Exception as e:
        print(f"Błąd przy zapisie do pliku {csv_path.as_posix()}: {e}")

def load_datasets():
    # Automatycznie załaduj wszystkie pliki .arff z folderu ../data/datasets/
    data_dir = Path("../data/datasets/")
    data_paths = []

    # Sprawdzenie czy folder istnieje
    if not data_dir.is_dir():
        print(f"Błąd: Folder {data_dir.as_posix()} nie istnieje!")
        return []

    # Znalezienie wszystkich plików .arff w folderze
    data_paths = [
        p.as_posix() for p in sorted(data_dir.glob("*.arff")) if p.is_file()
    ]

    print(f"Znaleziono {len(data_paths)} plików .arff w folderze {data_dir.as_posix()}")
    for i, path in enumerate(data_paths):
        print(f"  {i+1}. {path}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # data frames objects
    dfs = []
    for path in data_paths:
        df_temp = pd.DataFrame(arff.loadarff(path)[0])
        dfs.append(df_temp)
    print(f"\nnumber of loaded dataframes: {len(dfs)}")

    # preview dataframes
    for i, df in enumerate(dfs):
        print(f"podgląd zbioru nr {i +1}:")
        print(df.head())
    return data_paths

if __name__ == "__main__":
    start_time = datetime.now()

    # Saving paths to datasets
    datasets_paths = load_datasets()

    # Main tests loop
    all_results = []

    for dataset in datasets_paths:
        # definicja modelu ARF (Adaptive Random Forest classifier)
        rf_model = forest.ARFClassifier(n_models=10, seed=42)

        one_test_results = evaluate_stream(rf_model, dataset)
        all_results.append(one_test_results)

    # ONE DATASET TEST
    # rf_model = forest.ARFClassifier(n_models=10, seed=42)
    # one_test_results = evaluate_stream(rf_model, datasets_paths[3])
    # all_results.append(one_test_results)

    save_final_results(all_results)
    end_time = datetime.now()
    print(f"\nCzas wykonania: {end_time - start_time}")
