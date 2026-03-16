# 9 - Prescriptive analytics

## Riassunto dettagliato con spiegazione

## 1) Idea centrale
La prescriptive analytics è la fase in cui, dati insight descrittivi/predittivi, si decide l’azione ottima sotto vincoli.

Nucleo matematico: **ottimizzazione** (min/max di una funzione obiettivo su insieme ammissibile).

---

## 2) Formalizzazione del problema
Schema classico richiamato:
- data funzione `f: A -> R`,
- trovare `x0 in A` che minimizza o massimizza `f`.

Questa struttura astratta unifica moltissimi problemi reali (logistica, pianificazione, allocazione).

---

## 3) Principali sottocampi
Le slide passano in rassegna:
- linear programming (LP),
- nonlinear programming,
- quadratic programming,
- integer/mixed-integer programming (ILP/MILP),
- combinatorial optimization,
- stochastic/robust programming,
- constraint programming.

Messaggio didattico:
- il tipo di modello dipende da forma obiettivo, vincoli e natura delle variabili.

---

## 4) Complessità computazionale
Viene richiamato il tema P vs NP e la difficoltà di molte istanze reali.

Conseguenza pratica:
- ottimo esatto spesso non raggiungibile in tempo utile;
- serve compromesso qualità/tempo.

---

## 5) Dall’ottimo al “buono abbastanza”
La lezione enfatizza:
- in applicazioni industriali si cercano soluzioni subottime ma affidabili e rapide.

Questa è una posizione ingegneristica corretta: minimizzare costo decisionale totale, non solo gap matematico.

---

## 6) Ottimizzazione combinatoria
Struttura richiamata:
- insieme componenti `C`,
- soluzioni come sottoinsiemi `S` con vincoli di fattibilità,
- funzione costo `z(S)` da minimizzare.

È il pattern dietro routing, scheduling, assignment e planning.

---

## 7) Metaeuristiche
La parte finale introduce metodi di ricerca avanzata per problemi difficili.

In particolare:
- **VND (Variable Neighborhood Descent)**,
- **VNS (Variable Neighborhood Search)**,
- **ALNS (Adaptive Large Neighborhood Search)**.

---

## 8) VND/VNS: intuizione
- VND: esplora neighborhood diverse in ordine e resetta quando migliora.
- VNS: aggiunge perturbazione + accettazione per evitare stagnazione locale.

Significato:
- un ottimo locale in una neighborhood può non esserlo in un’altra;
- cambiare sistematicamente intorno di ricerca aumenta robustezza.

---

## 9) ALNS
ALNS alterna operatori di destroy/repair e aggiorna dinamicamente i pesi in base alle performance.

Perché è potente:
- apprende durante la ricerca quali mosse funzionano meglio;
- ottimo per problemi grandi e altamente vincolati.

---

## 10) Collegamento con predictive analytics
Messaggio di sistema del corso:
- predictive dice “cosa accadrà”;
- prescriptive dice “cosa conviene fare”.

Pipeline completa:
forecast -> scenari -> ottimizzazione -> piano operativo.

---

## 11) Conclusione didattica
La prescriptive analytics chiude il ciclo decisionale quantitativo.

In un progetto magistrale di ingegneria informatica, è il passaggio che trasforma il modello da analisi a **decision support operativo**, con vincoli reali, costi e trade-off espliciti.
