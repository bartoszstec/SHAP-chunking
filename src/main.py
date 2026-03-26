import pandas as pd
from scipy.io import arff
from river import stream, forest, metrics

# data paths to data containing obvious concept drift
data_paths = [
    '../data/Agrawal_f_1_2_p_5000_w_1_s_10000_r_7521.arff',
    '../data/Agrawal_f_2_3_p_5000_w_1_s_10000_r_7110.arff',
    '../data/Agrawal_f_3_4_p_5000_w_1_s_10000_r_5714.arff'
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

#definicja modelu ARF (Adaptive Random Forest classifier)
model = forest.ARFClassifier(n_models=10, seed=42)
metric = metrics.Accuracy() # wyświetlenie metryk
# print(f"Rozpoczynam proces dla: {data_paths[0]}")
# dataset = stream.iter_arff(data_paths[0], target='class')

# 1. Pobierz małą próbkę danych (np. pierwsze 100 rekordów)
dataset_init = stream.iter_arff(data_paths[0], target='class')
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
    y_clean = y.decode('utf-8') if isinstance(y, bytes) else y

    try:
        model.learn_one(x, y_clean)
    except TypeError as e:
        print(f"Błąd w rekordzie {i} przy wartości y={y} (typ: {type(y)}): {e}")

print("Model rozgrzany.")

for i, (x, y) in enumerate(dataset_init):
    # KROK A: Predykcja (Test)
    # y_pred to wynik, którego model "się domyśla" przed zobaczeniem poprawnej odpowiedzi
    y_pred = model.predict_one(x)

    # KROK B: Aktualizacja metryki
    if y_pred is not None:
        metric.update(y, y_pred)

    # KROK C: Nauka (Train)
    # Teraz model dostaje poprawną odpowiedź y i koryguje swoje wagi/drzewa
    model.learn_one(x, y)

    # KROK D: Podgląd postępów co 1000 rekordów
    if i % 1000 == 0:
        print(f"Rekord: {i} | Aktualne Accuracy: {metric.get():.4f}")

print(f"\nKońcowe Accuracy: {metric.get():.4f}")

