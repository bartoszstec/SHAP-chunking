import pandas as pd
from scipy.io import arff
from river import stream, forest, metrics
from river import drift

def evaluate_stream(model, dataset_path):
    # Inicjalizacja detektorów
    d_adwin = drift.ADWIN()
    d_ddm = drift.binary.DDM()
    d_pht = drift.PageHinkley()

    metric = metrics.Accuracy()  # wyświetlenie metryk

    # 1. Pobierz małą próbkę danych (np. pierwsze 100 rekordów)
    dataset_init = stream.iter_arff(dataset_path, target='class')
    # podebranie 100 pierwszych rekordów z datasetu i zapis do zwykłej tablicy
    initial_data = [next(dataset_init) for _ in range(100)]

    # 2. Wstępne trenowanie (Warm Start) z czyszczeniem
    print("Rozpoczynam rozgrzewanie modelu...")

    for i, (x, y) in enumerate(initial_data):
        # DIAGNOSTYKA: Sprawdźmy co jest w y
        if y is None:
            print(f"Uwaga: Rekord {i} ma pustą etykietę (None)!")
            continue

        # Naprawa typów (jeśli y to bajty b'0' lub b'1')
        y = y.decode('utf-8') if isinstance(y, bytes) else y

        try:
            model.learn_one(x, y)
        except TypeError as e:
            print(f"Błąd w rekordzie {i} przy wartości y={y} (typ: {type(y)}): {e}")

    print("Model rozgrzany.")

    drifts_found = {"ADWIN": [], "DDM": [], "PHT": []}
    for i, (x, y) in enumerate(dataset_init):

        # Naprawa typów (jeśli y to bajty b'0' lub b'1')
        y = y.decode('utf-8') if isinstance(y, bytes) else y

        # KROK 1: Predykcja (Test)
        # y_pred to wynik, którego model "się domyśla" przed zobaczeniem poprawnej odpowiedzi
        y_pred = model.predict_one(x)

        # KROK 2: Aktualizacja metryki
        if y_pred is not None:
            metric.update(y, y_pred)

            # Czy jest błąd?
            error = 0 if y_pred == y else 1

            # Aktualizacja detektorów
            d_adwin.update(error)
            d_ddm.update(True if error == 1 else False)
            d_pht.update(error)
            # Zamiast wrzucać do ADWIN-a informację o błędzie klasyfikacji (0 lub 1)
            # wrzucić wartości SHAP dla konkretnej cechy lub wektor ważności cech


            # Sprawdzanie dryftu
            if d_adwin.drift_detected:
                print(f"ADWIN wykrył dryft w rekordzie {i + 100}")
                drifts_found["ADWIN"].append(i + 100)

            if d_ddm.drift_detected:
                print(f"DDM wykrył dryft w rekordzie {i + 100}")
                drifts_found["DDM"].append(i + 100)

            if d_pht.drift_detected:
                print(f"PHT wykrył dryft w rekordzie {i + 100}")
                drifts_found["PHT"].append(i + 100)


        # KROK 3: Nauka (online training) - model aktualizuje swoją wiedzę na podstawie pojedynczego przykładu.
        # Teraz model dostaje poprawną odpowiedź y i koryguje swoje wagi/drzewa
        model.learn_one(x, y)

        # KROK D: Podgląd postępów co 1000 rekordów
        if i % 1000 == 0:
            print(f"Rekord: {i} | Aktualne Accuracy: {metric.get():.4f}")

    print(f"\nKońcowe Accuracy: {metric.get():.4f}")
    print(drifts_found)

def load_datasets():
    # data paths to data containing obvious concept drift
    data_paths = [
        '../data/Agrawal_f_1_2_p_5000_w_1_s_10000_r_7521.arff',
        '../data/Agrawal_f_2_3_p_5000_w_1_s_10000_r_7110.arff',
        '../data/Agrawal_f_3_4_p_5000_w_1_s_10000_r_5714.arff',
        '../data/SEA_f_1_2_p_5000_w_1_s_10000_r_6516.arff',
        '../data/SEA_f_1_3_p_5000_w_1_s_10000_r_7974.arff',
        '../data/SEA_f_2_3_p_5000_w_1_s_10000_r_3126.arff',
        '../data/STAGGER_f_1_2_p_5000_w_1_s_10000_r_2788.arff',
        '../data/STAGGER_f_1_3_p_5000_w_1_s_10000_r_6346.arff',
        '../data/STAGGER_f_2_3_p_5000_w_1_s_10000_r_3019.arff'
        ]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # data frames objects
    dfs = []
    for path in data_paths:
        df_temp = pd.DataFrame(arff.loadarff(path)[0])
        dfs.append(df_temp)
    print(f"number of loaded dataframes: {len(dfs)}")

    # preview dataframes
    for i, df in enumerate(dfs):
        print(f"podgląd zbioru nr {i +1}:")
        print(df.head())
    return data_paths

#definicja modelu ARF (Adaptive Random Forest classifier)
rf_model = forest.ARFClassifier(n_models=10, seed=42)


datasets_paths = load_datasets()
evaluate_stream(rf_model, datasets_paths[0])

