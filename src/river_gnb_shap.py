"""Analityczne (dokładne) wartości SHAP dla modeli Naive Bayes z River.

Działa z każdym modelem, którego log P(x, c) jest sumą wkładów poszczególnych cech:
  - MixedNB (Gaussian dla cech numerycznych + Bernoulli dla nominalnych),
  - Pipeline(TransformerUnion(...), OneHotEncoder, GaussianNB),
  - czyste GaussianNB / BernoulliNB / MultinomialNB z River.

Idea: dla modelu addytywnego  f_c(x) = b_c + Σ_f g_fc(x_f)  wartości Shapleya przy maskowaniu
interwencyjnym (jak shap.maskers.Independent) mają postać zamkniętą:
    φ_fc(x) = g_fc(x_f) − E_bg[g_fc(X_f)]
Wkład g_fc wyznaczamy bez zaglądania do wnętrza modelu:
    g_fc(x_f) + const_fc = jll({f: x_f})_c − jll({})_c
Stała (np. wyrazy Bernoulliego "cecha nieobecna") jest identyczna dla x i dla tła,
więc znosi się przy odejmowaniu średniej z tła. Wynik jest dokładny, koszt liniowy.
"""
import numpy as np


def _jll_function(model):
    """Zwraca funkcję x -> {klasa: log P(x, c)} dla modelu lub pipeline'u River."""
    if type(model).__name__ == "Pipeline":
        steps = list(model.steps.values())
        pre, nb = steps[:-1], steps[-1]

        def jll(x):
            for s in pre:
                x = s.transform_one(x)
            return nb.joint_log_likelihood(x)
        return jll
    if hasattr(model, "joint_log_likelihood"):
        return model.joint_log_likelihood
    raise TypeError(f"Model {type(model).__name__} nie udostępnia joint_log_likelihood")


def _to_array(d, classes):
    # Klasa, której model jeszcze nie zna -> 0 (brak wkładu); nie psuje centrowania
    return np.array([d.get(c, 0.0) for c in classes], dtype=float)


def per_feature_loglik(model, rows, feature_names, classes):
    """G[n, M, C] = wkład każdej oryginalnej cechy do log P(x, c), z dokładnością do stałej per cecha.
    rows: lista słowników cech w oryginalnej postaci (jak w strumieniu)."""
    jll = _jll_function(model)
    empty = _to_array(jll({}), classes)
    G = np.zeros((len(rows), len(feature_names), len(classes)))
    for n, x in enumerate(rows):
        for j, f in enumerate(feature_names):
            if f in x:
                G[n, j] = _to_array(jll({f: x[f]}), classes) - empty
    return G, empty


def analytic_shap(model, X_rows, bg_rows, feature_names, classes, output="centered_logit"):
    """Zwraca (values, base_values).

    output:
      "centered_logit" – log P(x,c) − średnia po klasach; values (n, M, C).
                         Addytywne dla dowolnej liczby klas (logity softmaxa).
      "log_odds"       – tylko 2 klasy: log P(c1, x) − log P(c0, x); values (n, M).
    """
    G, empty = per_feature_loglik(model, X_rows, feature_names, classes)
    Gbg, _ = per_feature_loglik(model, bg_rows, feature_names, classes)
    mbg = Gbg.mean(axis=0)                       # (M, C)
    base = empty + mbg.sum(axis=0)               # f(x) = base + Σ φ  (C,)
    phi = G - mbg                                # (n, M, C)

    if output == "centered_logit":
        phi = phi - phi.mean(axis=2, keepdims=True)
        base = base - base.mean()
        return phi, np.tile(base, (len(X_rows), 1))

    if output == "log_odds":
        assert len(classes) == 2
        return phi[..., 1] - phi[..., 0], np.full(len(X_rows), base[1] - base[0])

    raise ValueError(output)