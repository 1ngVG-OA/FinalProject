# 6 - Statistics, descriptive

## Riassunto dettagliato con spiegazione

## 1) Scopo della lezione
La slide deck fornisce i mattoni minimi di statistica descrittiva necessari per analizzare dati in modo rigoroso prima di passare a inferenza e forecasting.

Obiettivo pratico:
- descrivere campioni e distribuzioni,
- scegliere misure adeguate alla natura della variabile,
- evitare interpretazioni scorrette dei dati.

---

## 2) Popolazione, campione, variabili
Concetti fondamentali:
- **popolazione**: insieme completo di interesse;
- **campione**: sottoinsieme osservato;
- **variabili**: caratteristiche misurate (quantitative/categoriali).

Tipologie richiamate:
- quantitative continue/discrete;
- categoriali nominali/ordinali.

---

## 3) Livelli di misura
Le slide distinguono:
- nominale,
- ordinale,
- intervallo,
- rapporto.

Perché conta:
- il livello di misura determina operazioni lecite (ordinare, sommare, fare rapporti, ecc.).

Errore classico evitato dalla lezione:
- trattare variabili ordinali come se fossero a intervallo senza giustificazione.

---

## 4) Parametri vs statistiche
- **parametri**: proprietà vere della popolazione (spesso ignote);
- **statistiche**: stime calcolate sul campione.

La statistica descrittiva non “dimostra” ipotesi: **organizza e sintetizza** informazioni.

---

## 5) Distribuzioni di frequenza
Vengono introdotti:
- distribuzioni assolute/relative;
- istogrammi per continue;
- bar chart per categoriali.

Significato didattico:
- il primo controllo qualità dei dati è visivo e distributivo (asimmetrie, code, outlier, multimodalità).

---

## 6) Misure di posizione
Le slide coprono le misure principali:
- media,
- mediana,
- moda,
- quantili/percentili.

Interpretazione:
- media è sensibile agli outlier;
- mediana è robusta;
- quantili descrivono bene la struttura complessiva della distribuzione.

---

## 7) Misure di dispersione
Tipicamente richiamate:
- range,
- varianza,
- deviazione standard,
- IQR (interquartile range).

Per forecasting/data analysis sono cruciali perché:
- quantificano variabilità;
- influenzano scaling, scelta metriche e interpretazione del rischio.

---

## 8) Distribuzioni probabilistiche di base
Nella parte finale compaiono distribuzioni notevoli:
- binomiale,
- binomiale negativa,
- Poisson.

Uso pratico:
- modellare conteggi/eventi con ipotesi diverse su frequenza e indipendenza.

---

## 9) Aspetto computazionale
Le slide mostrano implementazioni Python (`scipy.stats`) per PMF e visualizzazione.

Messaggio operativo:
- teoria e codice devono andare insieme;
- i grafici di distribuzione aiutano a validare assunzioni prima della modellazione.

---

## 10) Conclusione didattica
La statistica descrittiva è la base di qualsiasi pipeline seria:
- prima descrivi bene i dati,
- poi scegli modelli e test coerenti.

In un progetto magistrale, una buona analisi descrittiva iniziale riduce errori metodologici nelle fasi successive.
