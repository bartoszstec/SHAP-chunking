# ADWIN (Adaptive Windowing)
# działanie: Zamiast stałego okna (np. ostatnie 100 próbek), ADWIN trzyma okno W, które rośnie, gdy dane są stabilne
# Algorytm utrzymuje okno danych, które dynamicznie się kurczy lub rozszerza. Gdy średnie z dwóch pod-okien różnią się od siebie bardziej niż ustalony próg
# (oparty na nierówności Hoeffdinga), ADWIN uznaje, że nastąpiła zmiana rozkładu (drift).

# Modyfikacja okna danych na podstawie wartości SHAPA
# problem: znalezienie punktu podziału
# SHAP - informatywność danej próbki w rozbiciu na cechy
# wystąpienie data driftu nie musi powodować concept driftu
# meta chunk
# przeanalizować KSwin

# DDM (Drift Detection Method)
# działanie: Algorytm monitoruje prawdopodobieństwo błędu oraz odchylenie standardowe
# DDM zakłada, że wskaźnik błędu modelu powinien maleć lub stabilizować się wraz z napływem danych. Algorytm wylicza dwa progi:
#
#     Poziom ostrzegawczy (Warning): Model zaczyna buforować dane do ewentualnego douczania.
#
#     Poziom dryftu (Drift): Rozkład uległ zmianie, wymagana jest reindeksacja lub reset modelu.

# Page-Hinkley Test (PHT)
# działanie: PHT sumuje różnice między aktualnymi wartościami a ich średnią kroczącą, ale odejmuje od tego dopuszczalny "szum" (δ)
# PHT oblicza skumulowaną sumę różnic między aktualnymi obserwacjami a ich średnią, pomniejszoną o dopuszczalny margines błędu.
# Jeśli ta suma przekroczy określoną wartość progową, ogłaszany jest dryft. Jest bardzo czuły na nagłe zmiany (abrupt drift).
