import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(file_path):
    if not Path(file_path).exists():
        print(f"Błąd: Plik {file_path} nie istnieje!")
        return None
    return pd.read_csv(file_path)


def plot_metrics(df):
    # Implementacja wykresu zbiorczego metryk dla wszystkich zestawów danych
    # wykres zawiera metryki: uśrednione ze wszystkich datasetów FDR, TPR, R, D1, D2 osobno dla każdego detektora
    detectors = ['ADWIN', 'KSWIN', 'DDM', 'PHT']
    metrics = {
        'FDR': 'false_discovery_rate',
        'TPR': 'true_positive_rate',
        'R': 'R',
        'D1': 'D1',
        'D2': 'D2',
    }

    # Wyliczenie średnich (pomijając braki danych, np. gdy detektor nic nie wykrył)
    summary = pd.DataFrame(index=metrics.keys(), columns=detectors, dtype=float)
    for det in detectors:
        for metric_label, metric_suffix in metrics.items():
            col_name = f'{det}_{metric_suffix}'
            summary.loc[metric_label, det] = df[col_name].mean(skipna=True)

    # FDR/TPR/R są w skali 0-1, a D1/D2 to opóźnienia liczone w próbkach (rząd tysięcy),
    # więc rysujemy je na dwóch osobnych podwykresach, żeby oba były czytelne.
    rate_metrics = ['FDR', 'TPR', 'R']
    delay_metrics = ['D1', 'D2']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    summary.loc[rate_metrics].plot(kind='bar', ax=ax1, edgecolor='black', legend=False)
    ax1.set_title('Wskaźniki jakości detekcji (0-1)')
    ax1.set_xlabel('Metryka')
    ax1.set_ylabel('Wartość średnia')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.tick_params(axis='x', rotation=0)

    summary.loc[delay_metrics].plot(kind='bar', ax=ax2, edgecolor='black')
    ax2.set_title('Opóźnienia detekcji (w próbkach)')
    ax2.set_xlabel('Metryka')
    ax2.set_ylabel('Wartość średnia')
    ax2.legend(title='Detektor')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.tick_params(axis='x', rotation=0)

    fig.suptitle('Uśrednione metryki detekcji dla poszczególnych detektorów', fontsize=14)
    plt.tight_layout()

    Path('../data/graphs').mkdir(parents=True, exist_ok=True)
    plt.savefig('../data/graphs/metrics_summary.png')
    plt.close()


def plot_datasets(df):
    # Implementacja wykresu zbiorczego liczby detekcji względem każdego zestawu danych
    # wykres zawiera liczbę detekcji dla każdego zestawu danych osobno dla każdego detektora
    detectors = ['ADWIN', 'KSWIN', 'DDM', 'PHT']
    cols = [f'{det}_detections_number' for det in detectors]

    counts = df[cols].copy()
    counts.columns = detectors
    counts.index = df['Dataset'].str.replace('.arff', '', regex=False)

    ax = counts.plot(kind='bar', figsize=(16, 7), edgecolor='black')
    ax.set_title('Liczba detekcji dla każdego zestawu danych', fontsize=14)
    ax.set_xlabel('Zestaw danych')
    ax.set_ylabel('Liczba detekcji')
    ax.legend(title='Detektor')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()

    Path('../data/graphs').mkdir(parents=True, exist_ok=True)
    plt.savefig('../data/graphs/datasets_summary.png')
    plt.close()


def plot_drift_timeline(df, dataset_index=0):
    """Generuje oś czasu zdarzeń dla konkretnego zbioru danych."""
    row = df.iloc[dataset_index]
    dataset_name = row['Dataset']
    p = row['Drift_Point']
    w = row['Width_Drift']

    plt.figure(figsize=(15, 6))

    # 1. Zaznaczanie okna dryftu
    start_drift = p - w / 2
    end_drift = p + w / 2
    plt.axvspan(start_drift, end_drift, color='red', alpha=0.15, label='Okno dryftu (Width)')
    plt.axvline(p, color='red', linestyle='--', alpha=0.5, label='Punkt centralny (p)')

    # 2. Rysowanie zdarzeń dla każdego detektora
    detectors = ['ADWIN', 'KSWIN', 'DDM', 'PHT']
    colors = ['blue', 'green', 'orange', 'purple']

    for i, det in enumerate(detectors):
        col_name = f'{det}_detections'
        if col_name in df.columns and pd.notna(row[col_name]):
            events = [float(e) for e in str(row[col_name]).split('; ') if e.strip()]
            plt.scatter(events, [i] * len(events), label=det, color=colors[i], s=100, edgecolors='black')

    plt.yticks(range(len(detectors)), detectors)
    plt.ylim(-1, len(detectors))
    plt.xlim(0, 10000)  # Twoja liczba rekordów
    plt.title(f'Oś czasu detekcji dla: {dataset_name}', fontsize=14)
    plt.xlabel('Numer rekordu w strumieniu')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'../data/graphs/timeline_{dataset_name.split(".")[0]}.png')

if __name__ == "__main__":
    # Ścieżka do Twojego pliku wygenerowanego przez main.py
    FILE_PATH = "../data/results/drift_detection_results.csv"

    results_df = load_data(FILE_PATH)

    if results_df is not None:
        # 1. Wykres zbiorczy opóźnień
        plot_metrics(results_df)

        # 2. Wykres zbiorczy opóźnień
        plot_datasets(results_df)

        # 3. Wykres osi czasu dla pierwszego zbioru (możesz zmienić indeks)
        # Np. dla Agrawal Gradual lub SEA
        for i in range(len(results_df)):
            plot_drift_timeline(results_df, dataset_index=i)