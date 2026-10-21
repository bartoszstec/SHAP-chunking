"""Wizualizacja: dlaczego model powinien zapominać po dryfie koncepcji.

Na jednym strumieniu z nagłym dryfem równolegle działają trzy kopie modelu MixedNB (test-then-train):
  - bez zapominania (jak w main.py),
  - reset w prawdziwym punkcie dryfu (wyrocznia - górna granica),
  - reset po detekcji ADWIN na binarnym błędzie (wariant praktyczny).
Wykres: (a) kroczące accuracy każdej kopii, (b, c) udział cech w SHAP agregowanym dla modelu
bez zapominania i z resetem - model bez zapominania długo wyjaśnia decyzje starą koncepcją.
Wynik: ../data/graphs/forgetting/<zbiór>.png (nadpisywany).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river import stream, drift

from main import build_model, parse_filename, SHAP_WINDOW
from shap_explainer import StreamShapExplainer, aggregate_shap, build_shap_reference, decode_label

DATASETS = [
    "../data/datasets/STAGGER_f_2_1_p_5000_w_1_s_10000_r_2778.arff",
    "../data/datasets/Agrawal_f_6_1_p_5000_w_1_s_10000_r_5567.arff",
    "../data/datasets/SEA_f_4_2_p_5000_w_1_s_10000_r_8897.arff",
]
OUT_DIR = Path("../data/graphs/forgetting")
WARM_START = 100
ACC_WINDOW = 200  # okno kroczącego accuracy
TOP_FEATURES = 4  # pozostałe cechy sumowane do "pozostałe"

# Paleta kategoryczna (stała kolejność): warianty modelu i cechy nie dzielą kolorów
LANE_COLORS = {"none": "#2a78d6", "oracle": "#eb6834", "adwin": "#1baf7a"}
LANE_LABELS = {"none": "bez zapominania", "oracle": "reset w punkcie dryfu",
               "adwin": "reset po detekcji ADWIN"}
FEATURE_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER_COLOR = "#9a9a96"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"


def run_lanes(dataset_path, true_drifts):
    it = stream.iter_arff(dataset_path, target='class')
    initial_data = [next(it) for _ in range(WARM_START)]
    background_rows, feature_names, classes = build_shap_reference(initial_data)

    lanes = {}
    for name in LANE_COLORS:
        model = build_model()
        for x, y in initial_data:
            if y is not None:
                model.learn_one(x, decode_label(y))
        lanes[name] = {"model": model, "correct": [], "detections": [], "shap": []}
    # SHAP tylko dla wariantu bez zapominania i wyroczni; tło = rozgrzewka (P(X) się nie zmienia)
    explainers = {n: StreamShapExplainer(lanes[n]["model"], background_rows, feature_names, classes)
                  for n in ("none", "oracle")}
    adwin = drift.ADWIN()
    buffer = []

    for i, (x, y) in enumerate(it):
        y = decode_label(y)
        record = i + WARM_START
        buffer.append(x)
        explain_now = len(buffer) == SHAP_WINDOW

        for name, lane in lanes.items():
            if name == "oracle" and record in true_drifts:
                lane["model"] = build_model()
                explainers["oracle"].model = lane["model"]
            model = lane["model"]
            ok = model.predict_one(x) == y
            lane["correct"].append(ok)
            if name == "adwin":
                adwin.update(0 if ok else 1)
                if adwin.drift_detected:
                    lane["detections"].append(record)
                    lane["model"] = model = build_model()
            if explain_now and name in explainers:
                per_feature = aggregate_shap(explainers[name].explain(buffer))
                lane["shap"].append((record, per_feature))
            model.learn_one(x, y)

        if explain_now:
            buffer.clear()
    return lanes, feature_names


def rolling_accuracy(correct):
    c = np.cumsum(np.asarray(correct, dtype=float))
    acc = (c[ACC_WINDOW:] - c[:-ACC_WINDOW]) / ACC_WINDOW
    x = np.arange(ACC_WINDOW, len(correct)) + WARM_START
    return x, acc


def feature_shares(shap_rows):
    records = np.array([r for r, _ in shap_rows])
    mass = np.array([v for _, v in shap_rows])
    total = mass.sum(axis=1, keepdims=True)
    # Pusty model tuż po resecie -> SHAP = 0, udział niezdefiniowany
    shares = np.divide(mass, total, out=np.full_like(mass, np.nan), where=total > 0)
    return records, shares


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def mark_drifts(ax, true_drifts, label=False):
    for k, p in enumerate(true_drifts):
        ax.axvline(p, color=INK, linestyle="--", linewidth=1.2,
                   label="prawdziwy dryf" if label and k == 0 else None)


def plot(dataset_path):
    info = parse_filename(dataset_path)
    true_drifts = info.get('p', [])
    lanes, feature_names = run_lanes(dataset_path, set(true_drifts))
    stem = Path(dataset_path).stem

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1]})
    fig.patch.set_facecolor("white")

    # (a) Kroczące accuracy
    ax = axes[0]
    for name, lane in lanes.items():
        x, acc = rolling_accuracy(lane["correct"])
        ax.plot(x, acc, color=LANE_COLORS[name], linewidth=1.6, label=LANE_LABELS[name])
    for d in lanes["adwin"]["detections"]:
        ax.plot(d, 0.02, marker="v", markersize=8, color=LANE_COLORS["adwin"],
                markeredgecolor="white", markeredgewidth=1.5, clip_on=False, transform=ax.get_xaxis_transform())
    mark_drifts(ax, true_drifts, label=True)
    ax.set_ylabel(f"accuracy (okno {ACC_WINDOW})", color=INK)
    ax.set_title("(a) Kroczące accuracy - ▼ detekcje ADWIN", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    style_axis(ax)

    # Te same cechy i kolory w obu panelach SHAP: największy łączny udział w obu wariantach
    shares = {n: feature_shares(lanes[n]["shap"]) for n in ("none", "oracle")}
    importance = sum(np.nansum(s, axis=0) for _, s in shares.values())
    top = list(np.argsort(importance)[::-1][:TOP_FEATURES])
    rest = [j for j in range(len(feature_names)) if j not in top]

    titles = {"none": "(b) Udział cech w SHAP agregowanym - model bez zapominania",
              "oracle": "(c) Udział cech w SHAP agregowanym - model z resetem w punkcie dryfu"}
    for ax, name in zip(axes[1:], ("none", "oracle")):
        records, s = shares[name]
        for color, j in zip(FEATURE_COLORS, top):
            ax.plot(records, s[:, j], color=color, linewidth=1.8, marker="o", markersize=3,
                    label=feature_names[j])
        if rest:
            ax.plot(records, s[:, rest].sum(axis=1), color=OTHER_COLOR, linewidth=1.4,
                    linestyle=":", label=f"pozostałe ({len(rest)})")
        mark_drifts(ax, true_drifts)
        ax.set_ylim(0, 1)
        ax.set_ylabel("udział w SHAP", color=INK)
        ax.set_title(titles[name], loc="left", fontsize=11, color=INK)
        style_axis(ax)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=False, title="cecha")

    axes[-1].set_xlabel("rekord", color=INK)
    fig.suptitle(f"Znaczenie zapominania po dryfie koncepcji - {stem}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Krótkie podsumowanie liczbowe: średnie accuracy po dryfie
    first = true_drifts[0] - WARM_START if true_drifts else 0
    for name, lane in lanes.items():
        print(f"  {LANE_LABELS[name]:<36} accuracy po pierwszym dryfie: {np.mean(lane['correct'][first:]):.3f}")
    print(f"Zapisano: {out.as_posix()}")


if __name__ == "__main__":
    for path in DATASETS:
        print(f"\n{Path(path).name}")
        plot(path)
