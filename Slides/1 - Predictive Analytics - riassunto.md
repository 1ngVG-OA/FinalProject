# 1 - Predictive Analytics

## Riassunto dettagliato con spiegazione

## 1) Definizione e obiettivo
La predictive analytics usa dati storici per stimare eventi futuri e supportare decisioni presenti.

Nucleo concettuale:
- identificare relazioni tra variabili esplicative e variabile target;
- sfruttare tali relazioni per forecast quantitativi.

---

## 2) Perché serve in pratica
Le slide mostrano applicazioni manageriali tipiche:
- domanda prodotti/servizi;
- fabbisogno personale;
- material requirement e magazzino;
- ricavi/profitti/perdite per pianificazione investimenti.

Messaggio: la previsione non è un fine, ma uno strumento di pianificazione.

---

## 3) Orizzonti temporali
Vengono distinti:
- **breve termine** (giorni/settimane): scheduling operativo;
- **medio termine** (settimane/mesi): capacità e fabbisogni;
- **lungo termine** (mesi/anni): trend strategici.

Punto didattico: modello e metrica cambiano con l’orizzonte.

---

## 4) Proprietà dei forecast
Le slide insistono su due aspetti:
- i forecast sono spesso sbagliati;
- quindi vanno comunicati con dispersione/intervalli, non solo punto medio.

Inoltre:
- serie aggregate tendono a essere più prevedibili delle serie granulari;
- l’incertezza cresce con la distanza temporale.

---

## 5) Famiglie di modelli
Classificazione proposta:
- **soggettivi** (Delphi, opinioni esperti, sales force composite);
- **oggettivi** (modelli matematici/statistici/econometrici, serie storiche).

Spiegazione da tutor:
- i metodi soggettivi incorporano conoscenza tacita;
- i metodi oggettivi garantiscono tracciabilità e replicabilità;
- in azienda spesso conviene un approccio ibrido.

---

## 6) Ciclo dei modelli oggettivi
Tre fasi:
1. **specifica** (identificazione modello);
2. **fitting** (stima parametri);
3. **diagnosi/validazione** (coerenza dati-modello).

Questa sequenza è la struttura minima di una pipeline scientificamente corretta.

---

## 7) Causalità vs serie storiche
Le slide distinguono:
- modelli causali/econometrici (driver espliciti);
- modelli time series (dipendenza dal passato della stessa variabile).

Scelta pratica:
- se hai driver robusti e misurabili, causal model;
- se la dinamica interna della serie domina, modelli autoregressivi/stagionali.

---

## 8) Valutazione errori
Sono richiamati indicatori classici:
- BIAS/ME,
- MAD/MAE,
- MSE,
- RMSE,
- MAPE.

Interpretazione:
- MAE è robusta e interpretabile in unità originali;
- RMSE penalizza molto errori grandi;
- MAPE è intuitiva ma instabile con valori vicini a zero.

---

## 9) Campionamento e validazione
Concetto fondamentale della lezione:
- separare **train** e **test** (es. 70/30, 90/10), rispettando l’ordine temporale.

Le slide ricordano anche:
- overfitting su train è frequente;
- in serie temporali la validazione deve essere temporale (non shuffle casuale).

---

## 10) Conclusione didattica
La predictive analytics è una disciplina di compromesso tra:
- accuratezza del modello,
- robustezza fuori campione,
- utilità decisionale.

Nel progetto magistrale, il valore è dimostrare non solo “quanto predice”, ma **quanto è affidabile e operativo** il forecast.
