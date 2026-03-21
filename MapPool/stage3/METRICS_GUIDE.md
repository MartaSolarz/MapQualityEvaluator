# 📊 Przewodnik po metrykach dla imbalanced data

## Twój problem: Ekstremalny imbalance (2.27% vs 97.73%)

Masz **43 razy więcej** próbek negatywnych niż pozytywnych!  
➡️ Standardowe metryki (jak accuracy) są **bardzo mylące**.

---

## 📐 Confusion Matrix - podstawa wszystkiego

```
                    Predykcja
                YES         NO
Prawda  YES     TP          FN     ← Ile map przegapiamy?
        NO      FP          TN     ← Ile nie-map źle oznaczamy?
```

**Twój przykład** (LogisticRegression):
```
                    Predykcja
                YES         NO
Prawda  YES     10          4      ← 10 dobrze, 4 przegapione
        NO      0           586    ← 0 pomyłek, 586 dobrze
```

---

## 🎯 Podstawowe metryki

### 1️⃣ **Precision (Precyzja)**
```
Precision = TP / (TP + FP)
```

**Co to znaczy?**  
Z wszystkich próbek, które model **oznaczył jako YES**, ile to naprawdę mapy?

**Twój wynik: 1.0000 (100%)**
- Model oznaczył 10 próbek jako mapy
- Wszystkie 10 to NAPRAWDĘ mapy
- **Zero false alarms!** 🎯

**Dlaczego ważne?**  
- High precision = nie tracisz czasu na adnotowanie śmieci
- False positive = zmarnowany czas na adnotację "nie-mapy"

---

### 2️⃣ **Recall (Czułość)**
```
Recall = TP / (TP + FN)
```

**Co to znaczy?**  
Z wszystkich **prawdziwych map**, ile model wyłapał?

**Twój wynik: 0.7143 (71.43%)**
- Jest 14 prawdziwych map w validation set
- Model znalazł 10 z nich
- **Przegapił 4** (28.57%)

**Dlaczego ważne?**  
- High recall = nie gubisz pozytywnych przykładów
- Low recall = zostawiasz mapy statystyczne w zbiorze jako "NO"

---

### 3️⃣ **F1-Score (Średnia harmoniczna)**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Co to znaczy?**  
Balans między precision i recall. Karze za ekstrema (np. precision=1.0 ale recall=0.1).

**Twój wynik: 0.8333**
- Świetny balans!
- Idealny precision (1.0) + dobry recall (0.71)

**Dlaczego ważne?**  
- Główna metryka dla imbalanced data
- Jedna liczba zamiast dwóch (precision + recall)
- Używana do rankingu modeli

---

### 4️⃣ **Specificity (True Negative Rate)**
```
Specificity = TN / (TN + FP)
```

**Co to znaczy?**  
Z wszystkich **prawdziwych negatywów** (nie-map), ile model poprawnie odrzucił?

**Twój wynik: 1.0000 (100%)**
- Jest 586 nie-map w validation set
- Model poprawnie odrzucił wszystkie 586
- **Zero false positives!**

**Dlaczego ważne?**  
- High specificity = model nie "halucynuje" map gdzie ich nie ma
- Dla Ciebie bardzo ważne (97% danych to nie-mapy)

---

## 🚨 **ACCURACY - dlaczego go NIE używamy?**

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Co to znaczy?**  
Procent wszystkich poprawnych predykcji.

**Twój wynik: 0.9933 (99.33%)**
```
(10 + 586) / (10 + 586 + 0 + 4) = 596/600 = 99.33%
```

### ⚠️ **Dlaczego to MYLĄCE?**

**Przykład: "Głupi model"**
```python
# Model który ZAWSZE mówi "NO" (nie jest to mapa)
def stupid_model(x):
    return 0  # zawsze NO
```

**Confusion Matrix głupiego modelu:**
```
                    Predykcja
                YES         NO
Prawda  YES     0           14     ← przegapił WSZYSTKIE mapy!
        NO      0           586    ← ale dobrze oznaczył nie-mapy
```

**Metryki głupiego modelu:**
- **Accuracy: 97.67%** 😱 (586/600) - WOW, "świetny" model!
- **Precision: undefined** (dzieli przez 0)
- **Recall: 0.0%** (nie znalazł żadnej mapy!)
- **F1-score: 0.0** (bezużyteczny!)

**Widzisz problem?**  
Głupi model ma **97.67% accuracy** tylko dlatego, że 97.67% danych to klasa negatywna!  
Model nic nie robi, ale accuracy wygląda świetnie.

---

### 📊 Porównanie: Twój model vs Głupi model

| Metryka | Twój LogReg | Głupi model | Co widzimy?
|---------|-------------|-------------|-------------|
| **Accuracy** | 99.33% | 97.67% | Tylko 2% różnicy! 😱
| **F1-score** | 83.33% | 0.0% | Ogromna różnica! ✅
| **Recall** | 71.43% | 0.0% | Twój model znajduje mapy! ✅
| **Precision** | 100% | undefined | Twój model nie halucynuje! ✅

**Wniosek:**  
Accuracy daje fałszywe poczucie sukcesu dla imbalanced data!  
F1, Precision, Recall pokazują prawdę.

---

## 🎪 Zaawansowane metryki

### 5️⃣ **ROC-AUC (Area Under ROC Curve)**
```
Zakres: 0.0 - 1.0 (wyżej = lepiej)
Random guess = 0.5
```

**Co to znaczy?**  
Mierzy jak dobrze model **separuje klasy** przy różnych progach (thresholds).

**Twój wynik: 0.9861**
- Prawie idealny! (1.0 = perfect)
- Model bardzo dobrze rozróżnia mapy od nie-map
- Prawdopodobieństwa są dobrze skalibrowane

**Dlaczego ważne?**  
- Threshold-independent (nie zależy od 0.5)
- Pokazuje ogólną jakość predykcji prawdopodobieństw
- **Kluczowe dla active learning** (potrzebujesz dobrych `predict_proba`)

**Kiedy używać:**  
- Gdy koszt FP i FN jest podobny
- Gdy chcesz ocenić ogólną jakość modelu

---

### 6️⃣ **PR-AUC (Precision-Recall AUC)**
```
Zakres: 0.0 - 1.0 (wyżej = lepiej)
Random guess = % klasy pozytywnej (u Ciebie: 0.0227)
```

**Co to znaczy?**  
Area under Precision-Recall curve. **Lepsza niż ROC-AUC dla imbalanced data!**

**Twój wynik: 0.8446**
- Bardzo dobry! (baseline = 2.27%)
- **37x lepszy niż random** (0.8446 / 0.0227 = 37)

**Dlaczego ważne dla Ciebie?**  
- ROC-AUC może być mylący gdy masz dużo negatywów (97%)
- PR-AUC koncentruje się na **rzadkiej klasie pozytywnej**
- **Najlepsza metryka dla ekstremalnie imbalanced data**

**Kiedy używać:**  
- Gdy imbalance > 10:1 (u Ciebie 43:1!)
- Gdy care about pozytywna klasa (u Ciebie: mapy statystyczne)

---

## 📈 Jak interpretować swoje wyniki?

### Twój model (LogisticRegression):

```
✅ Precision: 1.0000  → Nie marnujesz czasu na false alarms
✅ Recall: 0.7143     → Znajdujesz 71% map (dobry start!)
✅ F1-score: 0.8333   → Świetny balans
✅ ROC-AUC: 0.9861    → Excellent separacja klas
✅ PR-AUC: 0.8446     → 37x lepszy niż random
✅ Specificity: 1.0   → Zero false positives
❌ Accuracy: 0.9933   → Ignoruj (mylące!)
```

---

## 🎯 Trade-offs: Precision vs Recall

### Możesz zmienić threshold (teraz: 0.5) żeby dostosować balance:

**Niższy threshold (np. 0.3):**
```
✅ Recall ↑ (więcej map znalezionych)
❌ Precision ↓ (więcej false alarms)
```
**Użyj gdy:** nie możesz przegapić żadnej mapy (critical application)

**Wyższy threshold (np. 0.7):**
```
✅ Precision ↑ (mniej false alarms)
❌ Recall ↓ (więcej przegapionych map)
```
**Użyj gdy:** czas na adnotację jest drogi (active learning)

**Twój optymalny threshold: 0.6082**
- Maksymalizuje F1-score
- Już niemal idealny (threshold=0.5 też świetny!)

---

## 🎓 Które metryki śledzić?

### **Podczas trenowania:**
1. **F1-score** - główna metryka do rankingu modeli
2. **PR-AUC** - ogólna jakość dla imbalanced data
3. **Precision** - czy nie marnujesz czasu?
4. **Recall** - czy nie gubisz map?

### **Podczas active learning:**
1. **Recall** - czy model uczy się znajdować więcej map?
2. **F1-score** - czy overall quality rośnie?
3. **PR-AUC** - czy model się poprawia?

### **❌ Ignoruj:**
- **Accuracy** - mylący dla imbalanced data!

---

## 💡 Quick Reference

| Pytanie | Metryka | Twój wynik |
|---------|---------|------------|
| Czy model nie halucynuje? | **Precision** | 100% ✅ |
| Czy model znajduje mapy? | **Recall** | 71% ✅ |
| Jaki ogólny performance? | **F1-score** | 83% ✅ |
| Czy prawdopodobieństwa są dobre? | **ROC-AUC** | 98.6% ✅ |
| Jak dobre dla imbalanced? | **PR-AUC** | 84.5% ✅ |
| ~~Jaki % poprawnych?~~ | ~~Accuracy~~ | ~~Ignoruj~~ ❌ |

---

## 🚀 Następne kroki

Po każdej iteracji active learning śledź:
- **Recall powinien rosnąć** (uczysz model na trudnych przypadkach)
- **F1 powinien rosnąć** (overall quality improvement)
- **Precision może lekko spaść** (ale to OK, jeśli recall rośnie szybciej)

**Cel:**
- F1 > 0.90
- Recall > 0.85 (znajdujesz 85%+ map)
- Precision > 0.90 (< 10% false alarms)

---

**TL;DR:**
- ✅ **F1, Precision, Recall** - Twoi przyjaciele
- ✅ **PR-AUC** - najlepsza dla imbalanced
- ❌ **Accuracy** - wróg dla imbalanced data
- 🎯 **Twój model jest świetny!** (F1=83%, Precision=100%)
