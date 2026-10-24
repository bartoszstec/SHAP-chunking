"""Detektor dryfu oparty na zagregowanych wartościach SHAP (artykuł: data/article/article.pdf).

Wersja strumieniowa Algorytmów 1 i 2: detektor dostaje jeden wektor S_agg (M,) na chunk,
tak jak ADWIN dostaje jedną wartość na rekord.
  - Algorytm 1: pierwsze B chunków = baseline (zakładany bez dryfu); progi (T_l, T_u) dla każdej
    cechy = wartości z posortowanego baseline'u na pozycjach alpha*(B-1) i (1-alpha)*(B-1),
    zaokrąglonych do najbliższej (przy małym B zaokrąglenie w dół dawałoby alarm już na baseline).
  - Algorytm 2: średnia S_agg z ostatnich W chunków (okno może obejmować chunki baseline'u,
    jak start = i - W + 1 w artykule); wyjście poza progi którejkolwiek cechy = dryf.
  - Wiele dryfów: po detekcji baseline jest wyznaczany od nowa z kolejnych B chunków
    (wspólnie dla wszystkich cech).
"""
from collections import deque

import numpy as np


class ShapChunkDetector:
    def __init__(self, B=4, W=2, alpha=0.001):
        assert 0 < W <= B, "Wymagane 0 < W <= B"
        assert 0 < alpha < 0.5, "Wymagane 0 < alpha < 0.5"
        self.B, self.W, self.alpha = B, W, alpha
        self.baseline = []               # wektory S_agg bieżącego baseline'u
        self.recent = deque(maxlen=W)    # ostatnie W wektorów S_agg
        self.lower = self.upper = None   # progi per cecha, None = trwa zbieranie baseline'u

    def update(self, s_agg):
        """s_agg: np.array (M,) dla jednego chunku. Zwraca indeksy cech, dla których wykryto dryf
        (pusta tablica = brak dryfu)."""
        s_agg = np.asarray(s_agg, dtype=float)
        self.recent.append(s_agg)

        if self.lower is None:
            self.baseline.append(s_agg)
            if len(self.baseline) == self.B:
                ref = np.sort(np.array(self.baseline), axis=0)
                self.lower = ref[round(self.alpha * (self.B - 1))]
                self.upper = ref[round((1 - self.alpha) * (self.B - 1))]
            return np.array([], dtype=int)

        avg = np.mean(self.recent, axis=0)
        drifted = np.flatnonzero((avg < self.lower) | (avg > self.upper))
        if drifted.size:
            self.baseline, self.lower, self.upper = [], None, None
            self.recent.clear()
        return drifted
