# SHAP-chunking

## Założenia projektu:
- Celem projektu jest zbadanie skuteczności różnych detektorów concept driftu w kontekście modeli uczenia maszynowego operujących na danych strumieniowych
- Projekt będzie obejmował implementację różnych detektorów concept driftu, takich jak ADWIN, KSWIN, DDM, Page-Hinkley
- Strategia trenowania modelu będzie oparta na podejściu warm start, gdzie model będzie trenowany na danych historycznych, a następnie będzie testowany na danych strumieniowych
- Główną biblioteką do implementacji modeli uczenia maszynowego będzie river, która jest specjalnie zaprojektowana do pracy z danymi strumieniowymi
- 

## Planowane kroki:
1) Zbadanie performance detektorów concept driftu w różnych modelach (np. Random Forest, XGBoost, LightGBM) - porównanie wyników
2) Zbadanie wpływu różnych parametrów detektorów concept driftu (np. poziom istotności, rozmiar okna) na skuteczność detekcji
3) Utworzenie metody wykrywania concept driftu przy użyciu SHAP values, możliwie przy wykorzystaniu podejścia insteniejących już detektorów concept driftu, takich jak ADWIN, KSWIN, DDM, Page-Hinkley
4) miary d1, d2
5) wynznaczenie wartości shapa dla poszczególnych chunków i ewentualna wizualizacja za pomoca beesworm, waterfall
6) lasso - narzędzie do sterowania rozmiarem okna (lub inne metody rankingowania cech)
7) czy shap agregowany pokrywa się z wynikami lasso