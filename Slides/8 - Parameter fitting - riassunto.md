# 8 - Parameter fitting

## Riassunto dettagliato con spiegazione

## 1) Problema generale
Dato un modello parametrico e dati osservati, si vogliono trovare i parametri che minimizzano l’errore di previsione.

La lezione inquadra il tema come problema di ottimizzazione.

---

## 2) Lessico e inquadramento
Termini equivalenti richiamati:
- parameter estimation,
- parameter fitting,
- curve fitting,
- regression analysis.

Idea unificante:
- definisci una loss nel dominio dei parametri,
- trovi il minimo (o massimo) della funzione obiettivo.

---

## 3) Funzioni di errore
Le slide elencano metriche fondamentali:
- BIAS/ME,
- MAD/MAE,
- MSE,
- RMSE,
- MAPE,
- altre misure di qualità (corr/minmax nell’esempio codice).

Interpretazione da tutor:
- la scelta della loss influenza il tipo di modello che ottimizzi;
- MSE penalizza outlier più di MAE;
- MAPE va usata con cautela vicino allo zero.

---

## 4) Caso base: regressione lineare
La lezione mostra il fitting lineare come esempio canonico:
- modello semplice,
- obiettivo MSE,
- soluzione analitica (in casi standard) o numerica.

Didatticamente è il ponte verso metodi più generali.

---

## 5) Ottimizzazione numerica
Quando non hai soluzione chiusa:
- usi metodi iterativi (gradient-based o derivative-free).

Le slide mettono in evidenza che, al cuore del fitting moderno, c’è sempre ottimizzazione su una superficie di errore.

---

## 6) Metaeuristiche nel fitting
Parte avanzata: uso di metodi globali per superfici difficili/non convesse.

In particolare viene trattata **PSO (Particle Swarm Optimization)**:
- particelle con posizione/velocità;
- memoria del best personale (`pbest`) e globale (`gbest`);
- aggiornamenti con coefficienti cognitivi/sociali e inerzia.

---

## 7) PSO: significato operativo
Perché PSO nel fitting:
- evita dipendenza forte da gradiente;
- esplora globalmente spazi complessi;
- utile per tuning dove la loss è irregolare o multimodale.

Trade-off:
- più robustezza globale,
- ma più costo computazionale e sensibilità ai parametri di swarm.

---

## 8) Parametri pratici PSO (dalle slide)
Indicazioni empiriche:
- numero particelle tipicamente 10–50;
- bilanciamento tra componente personale e globale (`C1`, `C2`);
- controllo velocità per evitare stagnazione o instabilità.

Le slide mostrano anche tracce evolutive (PSO trace) per capire convergenza/esplorazione.

---

## 9) Messaggio metodologico
Il fitting non è “solo calcolo”: è progettazione di tre elementi coerenti:
1. modello,
2. loss,
3. algoritmo di ottimizzazione.

Se uno dei tre è mal allineato al problema, anche il risultato finale peggiora.

---

## 10) Conclusione didattica
La lezione collega statistica, ML e ottimizzazione in modo molto concreto.

Per un progetto magistrale robusto:
- motiva la loss scelta,
- giustifica l’algoritmo di fitting,
- documenta convergenza e sensibilità dei parametri.
