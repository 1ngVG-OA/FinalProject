# 7 - Statistics, inferential

## Riassunto dettagliato con spiegazione

## 1) Obiettivo della lezione
La lezione introduce la statistica inferenziale: usare un campione per trarre conclusioni sulla popolazione con controllo esplicito dell’incertezza.

Domanda guida:
- quanto è plausibile che ciò che vedo nel campione valga anche oltre il campione?

---

## 2) Campionamento e distribuzione campionaria
Punto teorico cardine richiamato:
- ripetendo campionamenti, le medie campionarie seguono una distribuzione (normalità asintotica).

Conseguenza pratica:
- si possono costruire intervalli e test per supportare decisioni quantitative.

---

## 3) Distribuzione normale e z-score
Le slide ripassano:
- proprietà della gaussiana;
- regola empirica 68-95-99;
- standardizzazione via z-score.

Interpretazione:
- z-score misura quante deviazioni standard separano un valore dalla media.

---

## 4) Intervalli di confidenza
Viene mostrata la costruzione di CI (95%, 99%) per la media.

Significato corretto:
- non è “probabilità che il parametro sia nel singolo intervallo osservato”,
- ma procedura che, ripetuta, copre il parametro vero nel livello dichiarato.

---

## 5) Test d’ipotesi
Schema tipico:
1. definire `H0` e `H1`;
2. scegliere statistica test;
3. fissare livello di significatività `\alpha`;
4. calcolare p-value;
5. decidere se rifiutare o no `H0`.

Le slide insistono sul ruolo del caso (errore random) nelle decisioni inferenziali.

---

## 6) Errore di I e II tipo
Contesto implicito richiamato dalla logica di test:
- errore I tipo: rifiuto `H0` vera;
- errore II tipo: non rifiuto `H0` falsa.

Per un progetto ML/forecasting:
- scegliere `\alpha` e test senza contesto di costo può portare a decisioni fuorvianti.

---

## 7) Confronto tra modelli predittivi
Parte molto rilevante del corso:
- non basta confrontare medie di metriche;
- serve testare se differenze di errore sono statisticamente significative.

Viene presentato il **Diebold-Mariano test** per confronto forecasting models.

---

## 8) Diebold-Mariano (DM)
Uso:
- confrontare due modelli sulla stessa serie/out-of-sample;
- verificare se differenza delle loss medie è diversa da zero.

Nelle slide:
- esempio pratico MLP vs LSTM;
- interpretazione tramite statistica DM e p-value;
- decisione in base a soglia critica (es. `\pm 1.96` per `\alpha=0.05` a due code).

---

## 9) Confronti multipli
Vengono citati anche diagrammi di differenza critica (con test non parametrici tipo Wilcoxon-Holm) per ranking multipli.

Messaggio metodologico:
- quando confronti molti modelli, devi correggere per confronti multipli e rappresentare gruppi non significativamente diversi.

---

## 10) Conclusione didattica
L’inferenza statistica è ciò che trasforma un confronto empirico in una conclusione difendibile.

In una tesi/progetto magistrale:
- metriche + test d’ipotesi + intervalli = valutazione robusta e credibile.
