# 5 - Predictive machine learning models

## Riassunto dettagliato con spiegazione

## 1) Obiettivo della lezione
Le slide introducono modelli ML non neurali per forecasting, con focus su:
- Support Vector Regression (SVR),
- alberi/ensemble (Random Forest, cenni a boosting e feature selection).

Idea guida: usare modelli capaci di catturare non linearità senza passare necessariamente da reti profonde.

---

## 2) Da SVM a SVR
Richiamo iniziale:
- SVM classifica separando classi con iperpiano a margine massimo.

Estensione regressiva:
- SVR usa tubo `\epsilon`-insensitive;
- errori entro `\epsilon` sono tollerati;
- si minimizza complessità + violazioni (slack variables).

Interpretazione pratica:
- `\epsilon` controlla sensibilità;
- `C` bilancia fit vs regolarizzazione;
- kernel gestisce non linearità.

---

## 3) Kernel trick
Le slide spiegano la mappa in feature space:
- trasformazione implicita tramite kernel (lineare, RBF, ecc.);
- modello lineare nello spazio trasformato = modello non lineare nello spazio originale.

Per forecasting:
- utile quando la relazione lag-target non è lineare ma ha struttura regolare.

---

## 4) Alberi decisionali per regressione
I regression tree vengono introdotti come modelli a partizione dello spazio feature.

Caratteristiche:
- regole interpretabili;
- gestione naturale di interazioni e non linearità;
- rischio di overfitting se albero troppo profondo.

---

## 5) Random Forest
Concetto centrale:
- ensemble di alberi addestrati su bootstrapping + sottoinsiemi casuali di feature;
- previsione finale aggregata (media in regressione).

Pro evidenziati:
- robustezza,
- training veloce,
- buona gestione di dati rumorosi/mancanti,
- estendibilità online.

Contro:
- interpretabilità meno diretta di un singolo albero;
- feature fusion meno trasparente;
- possibile minor resa su campioni piccoli.

---

## 6) Forecast su serie temporali con ML tabellare
Messaggio metodologico implicito:
- anche con SVR/RF devi convertire la serie in dataset supervisionato a lag.

Pipeline:
1. costruzione feature lag,
2. split temporale train/test,
3. fit modello,
4. forecast (one-step o iterativo),
5. valutazione con metriche coerenti.

---

## 7) Feature selection
Le slide citano RFE (Recursive Feature Elimination) per identificare lag più informativi.

Valore didattico:
- riduce dimensionalità;
- migliora interpretabilità;
- può aumentare generalizzazione.

---

## 8) Confronto concettuale con modelli neurali/statistici
- rispetto ai modelli statistici: più flessibili su non linearità;
- rispetto alle NN: spesso più semplici da allenare/tunare su dataset medi;
- ma richiedono comunque progettazione accurata delle feature temporali.

---

## 9) Conclusione didattica
I modelli ML non neurali sono una componente forte del toolkit predittivo.

In un progetto magistrale ben costruito:
- vanno confrontati con baseline statistiche e neurali,
- con stesso split temporale e stesse metriche,
- motivando trade-off tra accuratezza, robustezza e interpretabilità.
