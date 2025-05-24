# Uczenie ze Wzmocnieniem: Q-learning (Taxi-v3) i Deep Q-Network "from scratch" (LunarLander-v3)

**Autor:** Marcin Przybylski
**Data:** 2 maja 2025

## Opis Projektu

Niniejszy projekt dotyczy uczenia ze wzmocnieniem (Reinforcement Learning - RL). Celem było praktyczne zapoznanie się z podstawowymi koncepcjami RL poprzez implementację, trening i analizę dwóch popularnych algorytmów na klasycznych środowiskach z biblioteki Gymnasium (następcy OpenAI Gym).

Projekt został podzielony na dwie główne części:
1.  **Rozwiązanie środowiska Taxi-v3 przy użyciu algorytmu Q-learning:** Środowisko to charakteryzuje się dyskretną przestrzenią stanów i akcji. Agent (taksówka) uczył się optymalnej strategii odbierania i dostarczania pasażera.
2.  **Rozwiązanie środowiska LunarLander-v3 przy użyciu algorytmu Deep Q-Network (DQN) implementowanego "from scratch":** Środowisko lądowania na Księżycu posiada ciągłą przestrzeń stanów. Zaimplementowano algorytm DQN, w którym funkcja wartości Q jest aproksymowana przez sieć neuronową. Implementacja została wykonana od podstaw przy użyciu biblioteki NumPy, aby lepiej zrozumieć wewnętrzne mechanizmy algorytmu, w tym pamięć powtórek (Experience Replay) i sieć docelową (Target Network).

## 📂 Zasoby Projektu

Prace zostały zrealizowane w środowisku Google Colaboratory.

## ⚙️ Metodologia

### 1. Przygotowanie Środowisk

* **Taxi-v3:**
    * Dyskretna przestrzeń stanów: 500.
    * Dyskretna przestrzeń akcji: 6 (ruchy w 4 kierunkach, podniesienie, wysadzenie pasażera).
    * Render mode: `rgb_array`.
* **LunarLander-v3:**
    * Ciągła, 8-wymiarowa przestrzeń stanów (pozycja x, y, prędkość x, y, kąt, prędkość kątowa, kontakt nogi lewej/prawej).
    * Dyskretna przestrzeń akcji: 4 (nic, odpalenie lewego silnika, głównego, prawego).
    * Render mode: `rgb_array`.

Dla obu środowisk przeprowadzono wstępną eksplorację i sprawdzono działanie agenta wykonującego losowe akcje.

### 2. Implementacja Algorytmów

#### 2.2.1 Q-learning (dla Taxi-v3)

* Wartości Q przechowywane w tablicy NumPy (500, 6), inicjalizowanej zerami.
* Strategia eksploracji: e-greedy, gdzie $\epsilon$ liniowo zanikało od 1.0 do 0.01.
* Aktualizacja wartości Q: $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_{a'}Q(S',a') - Q(S,A)]$.

#### 2.2.2 Deep Q-Network (DQN) "from scratch" (dla LunarLander-v3)

* **Sieć neuronowa (NumPy):**
    * Warstwa wejściowa: 8 neuronów.
    * Dwie warstwy ukryte z aktywacją ReLU: 256 i 128 neuronów.
    * Warstwa wyjściowa: liniowa, 4 neurony (po jednym dla każdej akcji).
    * Inicjalizacja wag: He (dla ReLU), Glorot/Xavier (dla liniowej).
* **Pamięć powtórek (Experience Replay):** Bufor `deque` o rozmiarze 100 000, przechowujący krotki (S, A, R, S', done).
* **Sieć docelowa (Target Network):** Osobna kopia sieci głównej, wagi synchronizowane co 1000 kroków.
* **Aktualizacja (Trening):** Co 4 kroki losowano minibatch (rozmiar 64) z pamięci. Obliczano docelowe wartości Q (y) i wartości Q przewidziane przez sieć główną. Błąd ($Q_{current} - Q_{target\_bp}$ z clippingiem do [-1, 1]) propagowany wstecznie.
* Strategia eksploracji: e-greedy z liniowym zanikiem $\epsilon$.
* Dodatkowo: zanik współczynnika uczenia (learning rate).

### 2.3 Proces Treningu

* **Q-learning (Taxi-v3):**
    * Liczba epizodów: 15 000.
    * Parametry: $\alpha=0.1$, $\gamma=0.9$, $\epsilon$ zanikające od 1.0 do 0.01 przez 80% epizodów.
* **DQN (LunarLander-v3):**
    * Liczba epizodów: 2 000.
    * Parametry: $\gamma=0.99$, rozmiar batcha = 64, $\epsilon$ zanikające od 1.0 do 0.01 przez ≈ 600 000 kroków, LR zanikające od $5 \times 10^{-4}$ do $1 \times 10^{-5}$ przez ≈ 600 000 kroków.
    * Sieć docelowa aktualizowana co 1000 kroków, trening sieci głównej co 4 kroki (po zebraniu 640 próbek).

Monitorowano sumę nagród w każdym epizodzie oraz (dla DQN) średni loss.

### 2.4 Ewaluacja Modeli

* **Metryki ilościowe:** Średnia nagroda w ostatnich 100 epizodach treningu. Dedykowana faza ewaluacji (kilka epizodów bez eksploracji $\epsilon$-greedy) z obliczeniem średniej nagrody, średniej liczby kroków i odsetka pomyślnie zakończonych epizodów.
* **Wizualizacje:** Wykresy średniej kroczącej nagrody (i loss dla DQN). Animacje GIF pokazujące działanie nauczonych agentów.

## 🛠️ Wykorzystane Narzędzia i Technologie

* Python
* Gymnasium
* NumPy
* Matplotlib
* PIL (Pillow)
* imageio

## 📊 Wyniki i Interpretacja

### 3.1 Q-learning (Taxi-v3)

* **Wyniki treningu:** Po 15 000 epizodów, średnia nagroda w ostatnich 100 epizodach ustabilizowała się na poziomie **7.12**. Trening zajął około 37 sekund.
* **Wykres uczenia:** Wyraźny wzrost średniej nagrody kroczącej, wskazujący na efektywną naukę.
* **Wyniki ewaluacji (3 epizody bez eksploracji):**
    * Średnia nagroda: **10.33**.
    * Odsetek sukcesu: **100%**.
    * Średnia liczba kroków: **10.67**.
* **Interpretacja:** Algorytm Q-learning doskonale poradził sobie ze środowiskiem Taxi-v3. Dyskretna i stosunkowo niewielka przestrzeń stanów pozwoliła na efektywne wypełnienie tablicy Q i znalezienie optymalnej polityki.

### 3.2 Deep Q-Network (LunarLander-v3)

* **Wyniki treningu:** Po 2 000 epizodów (ok. 209 tys. kroków), średnia nagroda w ostatnich 100 epizodach wyniosła **-68.32**. Trening trwał około 578 sekund (prawie 10 minut).
* **Wykresy uczenia:** Tendencja wzrostowa średniej nagrody kroczącej (choć pozostaje ujemna) oraz tendencja spadkowa średniego lossu, co wskazuje na proces uczenia.
* **Wyniki ewaluacji (3 epizody bez eksploracji):**
    * Średnia nagroda: **-132.40**.
    * Odsetek sukcesu (próg >190 pkt): **0%**.
* **Interpretacja:** Implementacja DQN "from scratch" działała poprawnie i agent wykazywał postępy. Jednak środowisko LunarLander jest znacznie bardziej złożone. Osiągnięcie dobrych wyników (średnia nagroda > 200) wymaga zazwyczaj znacznie dłuższego treningu i potencjalnie dalszego strojenia hiperparametrów. Wynik -68.32 po 2000 epizodów jest typowy dla początkowej fazy uczenia.

## 🏁 Podsumowanie i Wnioski

1.  **Skuteczność Q-learningu w środowiskach dyskretnych:** Q-learning okazał się bardzo efektywny i szybko zbieżny w środowisku Taxi-v3.
2.  **Konieczność i złożoność DQN dla środowisk ciągłych:** Dla LunarLander-v3 konieczne było zastosowanie DQN. Osiągnięcie wysokiej wydajności wymaga znacznie więcej czasu treningu i optymalizacji.
3.  **Wyzwania implementacji "from scratch":** Implementacja DQN od podstaw przy użyciu NumPy jest pouczająca, ale bardziej złożona i podatna na błędy niż korzystanie z gotowych bibliotek RL.
4.  **Znaczenie hiperparametrów i czasu treningu:** Wyniki dla DQN podkreślają kluczową rolę odpowiedniego doboru hiperparametrów oraz wystarczająco długiego czasu treningu.
5.  **Praktyczne aspekty pracy z Gymnasium:** Projekt wymagał dostosowania do API biblioteki `gymnasium` oraz obsługi zależności i konfiguracji środowiska.

---
