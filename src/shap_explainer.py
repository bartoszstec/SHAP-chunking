"""SHAP dla modeli strumieniowych River - shap.PermutationExplainer na predict_proba.

Metoda niezależna od modelu: explainer widzi tylko funkcję X -> P(c | x).
Cechy nominalne (napisy) są kodowane do liczb całkowitych, bo masker SHAP operuje na macierzy
liczbowej - funkcja modelu dekoduje je z powrotem przed predykcją.
Jeśli model ma predict_proba_many(DataFrame), cały pakiet zamaskowanych wierszy idzie jednym
wywołaniem; w przeciwnym razie wiersz po wierszu przez predict_proba_one (wolniej).
"""
import numbers

import numpy as np
import pandas as pd
import shap


class StreamShapExplainer:
    def __init__(self, model, background_rows, feature_names, classes,
                 n_permutations=5, max_background=50, seed=42):
        """model - obiekt modelu (aktualizowany w miejscu, explainer zawsze widzi jego bieżący stan).
        background_rows - lista słowników cech (np. dane z rozgrzewki) - tło maskera.
        n_permutations - liczba permutacji na próbkę (każda liczona w przód i wstecz)."""
        self.model = model
        self.feature_names = list(feature_names)
        self.classes = list(classes)
        self.n_permutations = n_permutations
        self.nominal = {f for f in self.feature_names
                        if not isinstance(background_rows[0][f], numbers.Number)}
        self.categories = {f: [] for f in self.nominal}  # kod -> wartość
        self.codes = {f: {} for f in self.nominal}       # wartość -> kod

        # Podpróbkowanie tła tutaj (stałe ziarno), a nie w maskerze - masker drukuje ostrzeżenie przy każdym zbiorze
        background = shap.utils.sample(self.encode(background_rows), max_background, random_state=seed)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        self.explainer = shap.PermutationExplainer(
            self._predict, masker, feature_names=self.feature_names, seed=seed)

    def _code(self, f, v):
        # Nowa kategoria (niewidziana w tle) dostaje kolejny kod
        if v not in self.codes[f]:
            self.codes[f][v] = len(self.categories[f])
            self.categories[f].append(v)
        return self.codes[f][v]

    def encode(self, rows):
        return np.array([[self._code(f, x[f]) if f in self.nominal else x[f]
                          for f in self.feature_names] for x in rows], dtype=float)

    def decode(self, X):
        df = pd.DataFrame(X, columns=self.feature_names)
        for f in self.nominal:
            df[f] = np.asarray(self.categories[f], dtype=object)[df[f].round().astype(int)]
        return df

    def _predict(self, X):
        """(n, M) zakodowane -> (n, C) prawdopodobieństwa w kolejności self.classes."""
        df = self.decode(X)
        if hasattr(self.model, "predict_proba_many"):
            return self.model.predict_proba_many(df, self.classes)
        out = np.zeros((len(df), len(self.classes)))
        for n, x in enumerate(df.to_dict("records")):
            p = self.model.predict_proba_one(x)
            out[n] = [p.get(c, 0.0) for c in self.classes]
        return out

    def explain(self, rows):
        """Zwraca SHAP (n_samples, M, n_classes) w przestrzeni prawdopodobieństw."""
        X = self.encode(rows)
        max_evals = self.n_permutations * (2 * len(self.feature_names) + 1)
        return self.explainer(X, max_evals=max_evals, silent=True).values


def aggregate_shap(values):
    """Zwraca (SHAP agregowany - skalar, średni |SHAP| per cecha - wektor M).
    values: (n_samples, M, n_classes). Średnia po próbkach i klasach = S_agg z artykułu (Eq. 2)."""
    per_feature = np.abs(values).mean(axis=(0, 2))
    return per_feature


def decode_label(v):
    return v.decode('utf-8') if isinstance(v, bytes) else v


def build_shap_reference(initial_data):
    """Tło SHAP = dane z rozgrzewki w oryginalnej postaci (słowniki).
    Pierwszy rekord .arff to ({}, None) - pomijamy puste rekordy, inaczej feature_names = []."""
    valid_data = [(x, decode_label(y)) for x, y in initial_data if x and y is not None]
    feature_names = list(valid_data[0][0].keys())
    classes = sorted({y for _, y in valid_data})
    background_rows = [x for x, _ in valid_data]
    return background_rows, feature_names, classes
