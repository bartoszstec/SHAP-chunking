import pandas as pd
from scipy.io import arff
from river import stream, forest, metrics
from river import drift
import os

# Auxiliary functions
def calculate_latency(drift_founds, point_drift, width_drift):
    # Koniec okna dryftu (punkt, od którego liczymy 'spóźnienie')
    drift_end = point_drift + (width_drift / 2)

    # Szukamy pierwszej detekcji, która wystąpiła PO rozpoczęciu dryftu (p - w/2)
    # Bo detekcja w trakcie trwania zmiany (gradual) też jest sukcesem!
    drift_start = point_drift - (width_drift / 2)
    valid_detections = [d for d in drift_founds if d >= drift_start]

    if not valid_detections:
        return None  # Brak wykrycia dryftu

    first_detection = min(valid_detections)

    # Opóźnienie względem końca okna dryftu
    # Jeśli wynik jest ujemny, oznacza to, że wykryliśmy dryft w trakcie jego trwania (super!)
    return first_detection - drift_end

def evaluate_stream(model, dataset_path):
    # Detectors definition
    d_adwin = drift.ADWIN()
    d_ddm = drift.binary.DDM()
    d_pht = drift.PageHinkley()
    d_kswin = drift.KSWIN()

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

            # Drift detectors actualization
            d_adwin.update(error)
            d_ddm.update(True if error == 1 else False)
            d_pht.update(error)
            d_kswin.update(error)
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

    # Detection statistics
    adwin_latency = calculate_latency(drifts_found['ADWIN'], point_drift, width_drift)
    kswin_latency = calculate_latency(drifts_found['KSWIN'], point_drift, width_drift)
    ddm_latency = calculate_latency(drifts_found['DDM'], point_drift, width_drift)
    pht_latency = calculate_latency(drifts_found['PHT'], point_drift, width_drift)

    adwin_all_detections = len(drifts_found["ADWIN"]) if drifts_found["ADWIN"] else None
    kswin_all_detections = len(drifts_found["KSWIN"]) if drifts_found["KSWIN"] else None
    ddm_all_detections = len(drifts_found["DDM"]) if drifts_found["DDM"] else None
    pht_all_detections = len(drifts_found["PHT"]) if drifts_found["PHT"] else None

    adwin_all_str = "; ".join(map(str, drifts_found["ADWIN"])) if drifts_found["ADWIN"] else ""
    kswin_all_str = "; ".join(map(str, drifts_found["KSWIN"])) if drifts_found["KSWIN"] else ""
    ddm_all_str = "; ".join(map(str, drifts_found["DDM"])) if drifts_found["DDM"] else ""
    pht_all_str = "; ".join(map(str, drifts_found["PHT"])) if drifts_found["PHT"] else ""

    # Returns a dictionary with results for this set
    return {
        'Dataset': dataset_name,
        'Drift_Point': point_drift,
        'Width_Drift': width_drift,
        'Samples_Number': samples_number,

        'ADWIN_all_events': adwin_all_str,
        'KSWIN_all_events': kswin_all_str,
        'DDM_all_events': ddm_all_str,
        'PHT_all_events': pht_all_str,

        'ADWIN_all_detections': adwin_all_detections,
        'KSWIN_all_detections': kswin_all_detections,
        'DDM_all_detections': ddm_all_detections,
        'PHT_all_detections': pht_all_detections,

        'ADWIN_latency': adwin_latency,
        'KSWIN_latency': kswin_latency,
        'DDM_latency': ddm_latency,
        'PHT_latency': pht_latency,

        'Ending_Accuracy': metric.get()
    }

def save_final_results(all_results_list):
    csv_filename = "../data/results/drift_detection_results.csv"

    # Tworzymy DataFrame ze wszystkich wyników naraz
    df_results = pd.DataFrame(all_results_list)

    # Lista kolumn, które powinny być liczbami całkowitymi
    int_columns = [
        'point_drift', 'Width_Drift',
        'ADWIN_real_detection', 'ADWIN_all_detections',
        'KSWIN_real_detection', 'KSWIN_all_detections',
        'DDM_real_detection', 'DDM_all_detections',
        'PHT_real_detection', 'PHT_all_detections'
    ]

    # Wymuszamy typ Int64 (przez duże I) - on obsługuje <null> i nie robi floatów
    for col in int_columns:
        if col in df_results.columns:
            df_results[col] = df_results[col].astype('Int64')

    # Zapis do pliku
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    # Sprawdzamy, czy plik już istnieje, żeby wiedzieć czy dodać nagłówek
    file_exists = os.path.isfile(csv_filename)

    df_results.to_csv(
        csv_filename,
        mode='a',  # 'a' oznacza append (dopisywanie)
        index=False,
        header=not file_exists  # dodaj nagłówek tylko jeśli plik jest tworzony po raz pierwszy
    )

    print(f"\nWszystkie wyniki zapisane poprawnie do: {csv_filename}")
    print(df_results)

def load_datasets():
    # Automatycznie załaduj wszystkie pliki .arff z folderu ../data/datasets/
    data_dir = '../data/datasets/'
    data_paths = []

    # Sprawdzenie czy folder istnieje
    if not os.path.exists(data_dir):
        print(f"Błąd: Folder {data_dir} nie istnieje!")
        return data_paths

    # Znalezienie wszystkich plików .arff w folderze
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.arff'):
            full_path = os.path.join(data_dir, filename)
            data_paths.append(full_path)

    print(f"Znaleziono {len(data_paths)} plików .arff w folderze {data_dir}")
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
