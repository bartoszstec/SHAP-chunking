import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Błąd: Plik {file_path} nie istnieje!")
        return None
    return pd.read_csv(file_path)


def plot_latency_comparison(df):
    """Generuje wykres słupkowy porównujący opóźnienie detektorów."""
    plt.figure(figsize=(14, 7))

    # Przekształcenie danych do formatu "long" dla Seaborn
    latency_cols = ['ADWIN_latency', 'KSWIN_latency', 'DDM_latency', 'PHT_latency']
    # Upewniamy się, że bierzemy tylko te kolumny, które istnieją
    existing_cols = [c for c in latency_cols if c in df.columns]

    df_melted = df.melt(id_vars=['Dataset'], value_vars=existing_cols,
                        var_name='Detector', value_name='Latency')

    # Czyszczenie nazw detektorów (usuwanie '_latency')
    df_melted['Detector'] = df_melted['Detector'].str.replace('_latency', '')

    sns.barplot(data=df_melted, x='Dataset', y='Latency', hue='Detector')

    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title('Porównanie opóźnienia detekcji (Latency)\nUjemne wartości = wykrycie w trakcie trwania dryftu',
              fontsize=14)
    plt.ylabel('Opóźnienie (liczba rekordów)')
    plt.xlabel('Zbiór danych')
    plt.legend(title='Detektor')
    plt.tight_layout()
    plt.savefig('../data/graphs/latency_comparison.png')
    plt.show()

def plot_d1_d2_comparison(df):
    """Generuje dwa wykresy słupkowe porównujące metryki D1 i D2 detektorów oddzielnie."""

    # ========== WYKRES D1 ==========
    plt.figure(figsize=(14, 7))

    # Lista kolumn D1
    d1_cols = ['ADWIN_D1', 'KSWIN_D1', 'DDM_D1', 'PHT_D1']
    existing_d1_cols = [c for c in d1_cols if c in df.columns]

    df_d1_melted = df.melt(id_vars=['Dataset'], value_vars=existing_d1_cols,
                           var_name='Detector', value_name='D1_Value')

    # Czyszczenie nazw detektorów
    df_d1_melted['Detector'] = df_d1_melted['Detector'].str.replace('_D1', '')

    sns.barplot(data=df_d1_melted, x='Dataset', y='D1_Value', hue='Detector')

    plt.xticks(rotation=45, ha='right')
    plt.title('Porównanie metryki D1 detektorów\nNiższe wartości = lepsza detekcja',
              fontsize=14)
    plt.ylabel('Wartość D1 (średnia odległość od każdej detekcji do najbliższego prawdziwego dryftu)')
    plt.xlabel('Zbiór danych')
    plt.legend(title='Detektor')
    plt.tight_layout()
    plt.savefig('../data/graphs/d1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ========== WYKRES D2 ==========
    plt.figure(figsize=(14, 7))

    # Lista kolumn D2
    d2_cols = ['ADWIN_D2', 'KSWIN_D2', 'DDM_D2', 'PHT_D2']
    existing_d2_cols = [c for c in d2_cols if c in df.columns]

    df_d2_melted = df.melt(id_vars=['Dataset'], value_vars=existing_d2_cols,
                           var_name='Detector', value_name='D2_Value')

    # Czyszczenie nazw detektorów
    df_d2_melted['Detector'] = df_d2_melted['Detector'].str.replace('_D2', '')

    sns.barplot(data=df_d2_melted, x='Dataset', y='D2_Value', hue='Detector')

    plt.xticks(rotation=45, ha='right')
    plt.title('Porównanie metryki D2 detektorów\nNiższe wartości = lepsza detekcja',
              fontsize=14)
    plt.ylabel('Wartość D2 (średnia odległość od każdego prawdziwego dryftu do najbliższej detekcji)')
    plt.xlabel('Zbiór danych')
    plt.legend(title='Detektor')
    plt.tight_layout()
    plt.savefig('../data/graphs/d2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


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
        col_name = f'{det}_all_events'
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
        plot_latency_comparison(results_df)

        # 2. Wykres zbiorczy opóźnień
        plot_d1_d2_comparison(results_df)

        # 3. Wykres osi czasu dla pierwszego zbioru (możesz zmienić indeks)
        # Np. dla Agrawal Gradual lub SEA
        for i in range(len(results_df)):
            plot_drift_timeline(results_df, dataset_index=i)