import random
from river import drift

configs = {
    "default":      dict(delta=0.002, clock=32,  max_buckets=5, min_window_length=5,   grace_period=20),
    "A_nadwrazliwa":dict(delta=0.5,   clock=1,   max_buckets=5, min_window_length=2,   grace_period=5),
    "B_zachowawcza":dict(delta=1e-7,  clock=512, max_buckets=2, min_window_length=200, grace_period=2000),
}

# strumień: dryf nagły w t=1000 (0.2→0.6) i łagodny w t=3000 (0.6→0.45)
rng = random.Random(42)
def stream(n=5000):
    for t in range(n):
        p = 0.2 if t < 1000 else (0.6 if t < 3000 else 0.45)
        yield int(rng.random() < p)   # np. wskaźnik błędu klasyfikatora

data = list(stream())

for name, params in configs.items():
    det = drift.ADWIN(**params)
    detections = [t for t, x in enumerate(data) if (det.update(x) or det.drift_detected)]
    print(f"{name:15s} liczba detekcji={len(detections):3d}  pierwsze: {detections[:8]}")