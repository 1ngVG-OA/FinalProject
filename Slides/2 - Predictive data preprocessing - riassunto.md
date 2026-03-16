# 2 - Predictive data preprocessing

## Riassunto dettagliato con spiegazione

## 1) Obiettivo del preprocessing
Le slide spiegano che molti algoritmi richiedono proprietà specifiche dei dati.

Il preprocessing serve a rendere la serie:
- più compatibile con il modello scelto;
- più vicina a condizioni “ideali” di apprendimento (stabilità, scala, struttura interpretabile).

---

## 2) Momenti fondamentali e stazionarietà
Viene richiamata la descrizione statistica tramite:
- media,
- varianza,
- autocovarianza/autocorrelazione.

Serie stazionaria (in senso pratico):
- media stabile nel tempo,
- varianza stabile,
- struttura di dipendenza che non cambia con la traslazione temporale.

Nota didattica:
- molti modelli classici (ARMA, ecc.) funzionano bene solo se questa condizione è almeno approssimata.

---

## 3) Trasformazioni di base
Le tecniche principali mostrate:
- **differencing** (rimuove trend/parte di non-stazionarietà),
- **power transforms** (log, Box-Cox: stabilizzazione varianza),
- **standardizzazione/scaling** (soprattutto utile per modelli ML/NN).

Interpretazione:
- differencing agisce sulla dinamica;
- log/Box-Cox sulla distribuzione;
- scaling sulla numerica dell’ottimizzazione.

---

## 4) Stagionalità e decomposizione
Tema forte delle slide:
- riconoscere trend, stagione e residuo;
- eventualmente destagionalizzare prima del modello;
- ricostruire poi in post-processing sul dominio originale.

Punto pratico:
- sbagliare la gestione della stagionalità porta a errori sistematici anche con modelli “potenti”.

---

## 5) Verifiche statistiche utili
Vengono richiamati strumenti tipici:
- ACF/PACF per struttura temporale;
- test di stazionarietà (es. ADF nella pratica comune);
- controllo dei residui dopo trasformazione.

Messaggio da tutor:
- preprocess non è cosmetico: va validato quantitativamente.

---

## 6) Pipeline operativa corretta
Schema suggerito implicitamente dalle slide:
1. split temporale train/test;
2. fit trasformazioni solo su train (evitare leakage);
3. applicazione a val/test;
4. training modello;
5. inversione trasformazioni su forecast;
6. valutazione finale su scala originale.

---

## 7) Kalman filter (cenno avanzato)
Le ultime slide introducono il filtro di Kalman come metodo di smoothing/stima stato in presenza di rumore.

Concetti chiave:
- ciclo **predict -> update**;
- uso del **Kalman gain** per bilanciare fiducia tra previsione e misura;
- riduzione rumore e stima più stabile.

Perché è rilevante nel corso:
- è ponte tra preprocessing, filtraggio e modellazione dinamica.

---

## 8) Errori comuni
- fare trasformazioni dopo lo split in modo scorretto (leakage);
- valutare su scala trasformata e non su scala di business;
- ignorare ricostruzione/inversione di log-diff;
- standardizzare senza mantenere i parametri del train.

---

## 9) Conclusione didattica
Il preprocessing è parte integrante del modello, non un passaggio accessorio.

In una pipeline magistrale robusta:
- preprocess e modello vanno progettati insieme,
- e ogni trasformazione deve essere giustificata con evidenze (grafici/test/metriche).
