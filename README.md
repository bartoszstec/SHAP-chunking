# SHAP-chunking

## Spostrzeżenia badawcze:
1) Założenia strategii trenowania modelu: (konieczne określenie kierunku metodyki badawczej przed rozpoczęciem fazy testowej)
- Podejście **warm start**: trenowanie modelu na danych historycznych, w których concept drift nie występuje (bezpieczne dane), a następnie wpuszczenie modelu do danych strumieniowych
- Podejście **badawcze**: model uczy się wszystkiego od zera (w locie) - podejście wydaje się trudniejsze do zaimplementowania, ale jest bardziej "pure streaming"
2) Zamiast używać biblioteki scikit-learn prawdopodobnie zastosowana zostanie biblioteka river dająca więcej możliwości dla obsługi modeli operujących na danych strumieniowych