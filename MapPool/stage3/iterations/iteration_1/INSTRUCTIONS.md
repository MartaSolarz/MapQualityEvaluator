# Instrukcje adnotacji - Iteracja 1

## 📋 Zadanie
Zaadnotuj 600 **VALID** próbek jako:
- **YES** - mapa statystyczna
- **NO** - nie jest mapą statystyczną
- **INVALID** - URL nie działa lub błędny obraz (NIE LICZĄ SIĘ do targetu)

## 🎯 Cel
Active Learning wybrał te próbki jako **najbardziej niepewne** dla modelu.
Są to przypadki graniczne, które pomogą modelowi nauczyć się lepiej.

## 🔧 Jak adnotować?

### POLECANY: Użyj interface stage3 (automatycznie pomija invalid)

```bash
cd MapPool/stage3
streamlit run annotation_interface.py -- --iteration 1 --target 600
```

**Automatyczne funkcje:**
- ✅ HEAD request sprawdza URL przed wyświetleniem
- ✅ Automatycznie pomija invalid URLs (NIE liczą się do targetu!)
- ✅ Zapisuje progress po każdej adnotacji
- ✅ Resume: możesz przerwać i wrócić później
- ✅ Pokazuje model prediction dla każdej próbki

## 📊 Co zobaczysz?

- **Więcej map** niż w pierwszej rundzie (model teraz szuka trudnych przypadków)
- **Case'y graniczne:** mapy niskiej jakości, częściowe mapy, edge cases
- **Model prediction hint:** interfejs pokazuje co model myśli o próbce
- **Niepewne przypadki:** prawdopodobieństwa blisko 0.5

## ✅ Po adnotacji

Interface automatycznie zapisze wyniki do:
- `iteration_1/annotated.parquet` (gdy osiągniesz target)
- `iteration_1/annotations_progress.jsonl` (progress)

Następnie możesz:
1. Użyć `retrain_model.py` (TODO) żeby retrenować model
2. Uruchomić kolejną iterację active learning

## 🎲 Statystyki

Interface wybierze więcej próbek niż 600 (buffer na invalid URLs).
Oczekiwany % invalid: ~38%, więc wybranych zostanie ~967 próbek.
Ale adnotujesz tylko do osiągnięcia 600 VALID!
