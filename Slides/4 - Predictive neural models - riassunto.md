# 4 - Predictive neural models

## Riassunto dettagliato con spiegazione

## 1) Idea centrale
Le reti neurali sono presentate come modelli molto efficaci per forecasting non lineare.

Punti chiave:
- possono apprendere pattern lineari e non lineari;
- sono non-parametriche (non richiedono assunzioni forti sulla distribuzione del rumore);
- approssimano la funzione generatrice della serie.

---

## 2) Mappatura terminologica statistica-neurale
Le slide chiariscono l’analogia:
- input nodes -> variabili indipendenti (lag),
- output nodes -> variabile dipendente (target),
- weights -> parametri,
- training -> identificazione/stima.

Questo aiuta a collegare NN e modelli statistici classici.

---

## 3) Da serie temporale a dataset supervisionato
Passaggio cruciale:
- creare finestre scorrevoli (sliding window);
- ogni record contiene i lag passati e il target futuro.

È la base per usare MLP, RNN, LSTM su serie univariate.

---

## 4) Meccanismo di apprendimento
Ciclo mostrato:
1. finestra in input,
2. output predetto,
3. confronto col valore reale,
4. backpropagation,
5. shift finestra e ripetizione.

Interpretazione:
- è una procedura di minimizzazione errore su sequenze temporali.

---

## 5) Architetture richiamate
Le slide coprono principalmente:
- **MLP feedforward** (baseline neurale su lag),
- **RNN/LSTM** (memoria dinamica su dipendenze più lunghe).

Collegamento concettuale utile:
- MLP su lag ~ regressore autoregressivo non lineare;
- RNN/LSTM catturano dinamiche dipendenti dallo stato nel tempo.

---

## 6) Preprocessing e scaling
Viene evidenziato l’uso di `MinMaxScaler` prima del training.

Perché è importante:
- migliora stabilità numerica;
- accelera convergenza;
- evita gradienti mal condizionati.

---

## 7) Forecast multi-step
Le slide mostrano forecasting iterativo:
- si predice uno step;
- la previsione entra nella finestra successiva;
- si ripete per `h` periodi.

Nota didattica:
- questo schema introduce accumulo d’errore, quindi va valutato con attenzione su orizzonti più lunghi.

---

## 8) Implementazione pratica (PyTorch)
Elementi essenziali mostrati:
- definizione modello,
- loss `MSELoss`,
- ottimizzatore Adam,
- loop di training per epoche,
- fase `eval` con `torch.no_grad()`,
- inversione scaling per interpretazione reale.

---

## 9) Pro e contro (sintesi)
Vantaggi:
- alta capacità espressiva;
- buona resa su non linearità complesse.

Limiti:
- maggiore sensibilità a iperparametri;
- necessità di più dati/regularization;
- minore interpretabilità rispetto ai modelli statistici classici.

---

## 10) Messaggio metodologico finale
La lezione non propone le NN come sostituto universale, ma come famiglia complementare.

Approccio corretto in un progetto magistrale:
- baseline statistica forte,
- confronto equo con modelli neurali,
- validazione temporale rigorosa e diagnostica errori.
