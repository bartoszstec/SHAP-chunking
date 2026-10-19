import random
from pathlib import Path

import matplotlib.pyplot as plt
from river import drift

configs = {
    "default":      dict(delta=0.002, clock=32,  max_buckets=5, min_window_length=5,   grace_period=20),
    "A":dict(delta=0.5,   clock=1,   max_buckets=5, min_window_length=2,   grace_period=5),
    "B":dict(delta=1e-7,  clock=512, max_buckets=2, min_window_length=200, grace_period=2000),
}

# strumień: dryf nagły w t=1000 (0.2→0.6) i łagodny w t=3000 (0.6→0.45)
N_SAMPLES = 5000
DRIFT_POINTS = [1000, 3000]

rng = random.Random(42)
def stream(n=N_SAMPLES):
    for t in range(n):
        p = 0.2 if t < DRIFT_POINTS[0] else (0.6 if t < DRIFT_POINTS[1] else 0.45)
        yield int(rng.random() < p)   # np. wskaźnik błędu klasyfikatora


def plot_config_timeline(detections_per_config, drift_points, n_samples,
                         out_path='../data/graphs/adwin_configs_timeline.png'):
    """Oś czasu: rzeczywiste dryfy (pionowe linie) i detekcje każdej konfiguracji ADWIN (osobny wiersz)."""
    names = list(detections_per_config)

    plt.figure(figsize=(15, 1.2 * len(names) + 2))

    # 1. Rzeczywiste momenty dryfu (etykieta tylko raz, żeby legenda się nie powtarzała)
    for j, p in enumerate(drift_points):
        plt.axvline(p, color='red', linestyle='--', linewidth=2, alpha=0.7,
                    label='Rzeczywisty dryf' if j == 0 else None)

    # 2. Detekcje dla każdej konfiguracji detektora
    for i, name in enumerate(names):
        events = detections_per_config[name]
        plt.scatter(events, [i] * len(events), s=100, color='blue', edgecolors='black',
                    label='Detekcja' if i == 0 else None)

    plt.yticks(range(len(names)), names, fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylim(-1, len(names))
    plt.xlim(0, n_samples)
    plt.title('Momenty detekcji dryfu dla różnych konfiguracji ADWIN', fontsize=18)
    plt.xlabel('Numer próbki w strumieniu', fontsize=16)
    plt.ylabel('Konfiguracja detektora', fontsize=16)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=13)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Zapisano wykres: {out_path}")


data = list(stream())

detections_per_config = {}
for name, params in configs.items():
    det = drift.ADWIN(**params)
    detections = [t for t, x in enumerate(data) if (det.update(x) or det.drift_detected)]
    detections_per_config[name] = detections
    print(f"{name:15s} liczba detekcji={len(detections):3d}  pierwsze: {detections[:8]}")

plot_config_timeline(detections_per_config, DRIFT_POINTS, N_SAMPLES)
