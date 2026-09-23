from pathlib import Path
import pandas as pd
from scipy.io import arff
import numbers
from river import stream, naive_bayes, metrics, drift, compose, preprocessing
from D1D2_metrics import D1D2
from datetime import datetime

# Auxiliary functions
# R - adjusted ratio of the number of true drifts to the number of detections
def calculate_r(true_drifts_number, detected_drifts_number):
    if detected_drifts_number == 0:
        return None
    return abs((abs(true_drifts_number) / abs(detected_drifts_number)) - 1)

def build_model():
    return compose.Pipeline(
        compose.TransformerUnion(
            compose.SelectType(numbers.Number),
            compose.SelectType(str) | preprocessing.OneHotEncoder()
        ),
        naive_bayes.GaussianNB()
    )


# Extracting information from a file name
# Formula: DatasetName_f_F1_F2_p_P_w_W_s_S_r_R
# -------> F1, F2 - features used for drift, P - point of drift, W - width of drift, S - number of samples, R - random seed
def parse_filename(dataset_path):
    # Formula: Name_f_F1_F2..._p_P1_P2..._w_W1_W2..._s_S_r_R  (p i w opcjonalne)
    stem = Path(dataset_path).stem          # bez '.arff'
    parts = stem.split('_')
    markers = {'f', 'p', 'w', 's', 'r'}

    info = {'name': parts[0]}
    key = None
    for tok in parts[1:]:
        if tok in markers:
            key = tok
            info[key] = []
        elif key is not None:
            info[key].append(int(tok))
    return info

def detection_rates(true_drifts, detections, widths):
    """TPR i FDR z osobną szerokością okna dla każdego dryftu (bez modyfikacji D1D2)."""
    if not true_drifts:
        # Brak dryftu: każda detekcja to fałszywy alarm, TPR niezdefiniowane
        fdr = 1.0 if detections else None
        return None, fdr

    # Założenie: okna kolejnych dryftów się nie nakładają
    windows = [D1D2.get_window(tg, w) for tg, w in zip(true_drifts, widths)]
    for (_, end_prev), (start_next, _) in zip(windows, windows[1:]):
        assert end_prev < start_next, f"Okna dryftów się nakładają: {windows}"

    matched = int(sum(D1D2.tpr([tg], detections, w)
                      for tg, w in zip(true_drifts, widths)))

    tpr = matched / len(true_drifts)
    fdr = (len(detections) - matched) / len(detections) if detections else None
    return tpr, fdr

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
        proba = model.predict_proba_one(x) # Class probability
        true_class_proba = proba.get(y, 0.0)
        # print(f"\ny_pred: {y_pred}, \nproba: {proba}, \ntrue_class_proba: {true_class_proba}")

        # Actualization
        if y_pred is not None:
            metric.update(y, y_pred)

            # Classification error
            error = 0 if y_pred == y else 1

            # Drift detectors actualization
            d_adwin.update(error)                           # ADWIN
            #d_kswin.update(error)                           # KSWIN
            d_kswin.update(float(true_class_proba))       # KSWIN - alternative approach using class probability
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

    # Parse dataset filename to extract drift information
    info = parse_filename(dataset_path)
    dataset_name = Path(dataset_path).name
    true_drifts = info.get('p', [])  # list of all true drifts
    widths = info.get('w', [])
    samples_number = info['s'][0]
    # true_drifts_number = len(true_drifts) # number of true drifts


    # Returns a dictionary with results for this dataset
    results = {
        'Dataset': dataset_name,
        'Drift_Point': "; ".join(map(str, true_drifts)),
        'Width_Drift': "; ".join(map(str, widths)),
        'Samples_Number': samples_number,
    }

    for name, dets in drifts_found.items():
        tpr, fdr = detection_rates(true_drifts, dets, widths)
        r = calculate_r(len(true_drifts), len(dets)) if true_drifts else None
        d1 = D1D2.D1(true_drifts, dets)  # D1/D2 same zwracają None przy pustych listach
        d2 = D1D2.D2(true_drifts, dets)

        results[f'{name}_detections'] = "; ".join(map(str, dets))
        results[f'{name}_detections_number'] = len(dets)
        results[f'{name}_false_discovery_rate'] = round(fdr, 2) if fdr is not None else None
        results[f'{name}_true_positive_rate'] = round(tpr, 2) if tpr is not None else None
        results[f'{name}_R'] = round(r, 2) if r is not None else samples_number
        results[f'{name}_D1'] = round(d1) if d1 is not None else samples_number
        results[f'{name}_D2'] = round(d2) if d2 is not None else samples_number

    results['Ending_Accuracy'] = metric.get()

    return results

def save_final_results(all_results_list):
    csv_path = Path("../data/results/drift_detectors_results.csv")

    # Tworzymy DataFrame ze wszystkich wyników naraz
    df_results = pd.DataFrame(all_results_list)

    # Lista kolumn, które powinny być liczbami całkowitymi
    int_columns = [
        'Samples_Number',
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
        nb_model = build_model()

        one_test_results = evaluate_stream(nb_model, dataset)
        all_results.append(one_test_results)

    save_final_results(all_results)
    end_time = datetime.now()
    print(f"\nCzas wykonania: {end_time - start_time}")
