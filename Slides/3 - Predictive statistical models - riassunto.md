# 3 - Predictive statistical models

## Riassunto dettagliato con spiegazione (taglio da tutor magistrale)

## 1) Obiettivo della lezione
La slide deck presenta i principali modelli statistici classici per forecasting di serie temporali e mostra:
- quando usarli;
- quali assunzioni richiedono;
- come stimare e scegliere i parametri;
- come leggere diagnostica e qualità del fit.

Il messaggio centrale è: **il modello va scelto in base alla struttura della serie** (trend, stagionalità, esogene, stazionarietà), non in base alla sola popolarità dell’algoritmo.

---

## 2) Famiglie di metodi presentate
La lezione elenca i blocchi principali:
- AR, MA, ARMA
- ARIMA
- SARIMA / SARIMAX
- SES (Simple Exponential Smoothing)
- Holt-Winters (HWES)
- cenno a Theta

Interpretazione didattica:
- **AR/MA/ARMA**: base lineare su serie stazionarie senza trend/stagione.
- **ARIMA**: estende ARMA introducendo differenziazione (parte “I”) per gestire trend.
- **SARIMA/SARIMAX**: estende ARIMA per stagionalità (e con X per variabili esogene).
- **SES/Holt-Winters**: smoothing esponenziale, molto pratico e competitivo su molte serie reali.

---

## 3) AR(p): autoregressione
Forma concettuale:
\[
 y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \varepsilon_t
\]

Significato:
- il valore futuro dipende linearmente dai precedenti `p` lag;
- l’errore è una componente casuale.

Punti chiave delle slide:
- richiede (in pratica) **stazionarietà**;
- modella solo dipendenze lineari;
- utile su serie univariate senza trend/stagionalità marcata.

Nel “fil rouge” viene mostrato AR(2) su dati log-differenziati: è un esempio classico di pipeline corretta (trasformazione -> stima -> verifica).

---

## 4) AIC/AICc: come confrontare modelli
AIC è criterio informativo per confronto relativo:
\[
\mathrm{AIC} = -2\log L + 2k
\]
con `L` likelihood massima e `k` numero parametri.

Lettura:
- più basso = miglior compromesso fit/complessità;
- non è una metrica assoluta: serve per **confronto tra candidati** sullo stesso dataset.

Le slide evidenziano un punto cruciale:
- con campioni piccoli AIC può favorire overfitting;
- si usa quindi **AICc** (AIC corretto), che penalizza di più i modelli complessi.

Messaggio pratico:
- in time series model selection “seria”, AIC/AICc sono spesso il primo filtro, poi si passa a residual diagnostics e test out-of-sample.

---

## 5) LASSO nel contesto autoregressivo
LASSO (regolarizzazione L1) viene introdotto come:
- metodo di regressione con shrinkage;
- possibile strumento di selezione dei lag rilevanti.

Idea:
- trasformi la serie in problema supervisionato (matrice di lag come feature);
- L1 può azzerare coefficienti inutili.

Nota didattica importante:
- è ibrido tra statistica classica e approccio ML;
- molto utile quando i lag candidati sono tanti e vuoi sparsità interpretabile.

---

## 6) MA(q): media mobile sugli errori
Forma:
\[
 y_t = \mu + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j} + \varepsilon_t
\]

Interpretazione:
- non usa direttamente lag di `y_t`, ma lag dell’errore.

Nelle slide è sottolineato che MA puro si mostra in pratica tramite ARIMA con `p=0, d=0`.

---

## 7) ARMA(p,q)
Combina AR e MA:
\[
 y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j} + \varepsilon_t
\]

Quando usarlo:
- serie stazionaria;
- senza trend e senza stagionalità evidente.

Spiegazione del perché è utile:
- AR cattura memoria “strutturale” della serie;
- MA assorbe struttura residua degli errori;
- insieme spesso migliorano il fit rispetto a AR o MA singoli.

---

## 8) ARIMA(p,d,q): la parte “Integrated”
È il cuore del forecasting classico moderno.

Ruolo dei parametri:
- `p`: ordine autoregressivo;
- `d`: numero differenziazioni;
- `q`: ordine media mobile.

L’idea è modellare una serie resa stazionaria tramite differenze. Se `d=0`, ARIMA si riduce ad ARMA.

Punto metodologico forte della lezione: **Box-Jenkins**
1. Identificazione (trend/stagione/lag plausibili)
2. Stima parametri
3. Diagnostica residuale
4. Iterazione

Questa è la procedura corretta da presentare in un progetto magistrale, perché rende esplicito il ciclo scientifico (ipotesi -> test -> revisione).

---

## 9) Ricerca iperparametri
Le slide mostrano due idee:
- **grid search** su intervalli discreti di `(p,d,q)`;
- campionamento per parametri continui (citato Hammersley).

Messaggio pratico:
- i modelli sono sensibili ai parametri;
- serve una strategia sistematica, non tentativi casuali.

---

## 10) SARIMA: gestire la stagionalità
Notazione:
\[
\mathrm{SARIMA}(p,d,q)\times(P,D,Q)_m
\]

Dove:
- `(p,d,q)` è parte non stagionale;
- `(P,D,Q)` è parte stagionale;
- `m` è periodicità stagionale (es. 12 per mesi, 4 per trimestri).

Intuizione:
- ARIMA cattura dinamica locale;
- SARIMA aggiunge dinamica su cicli stagionali.

Le slide mostrano sia:
- `pmdarima.auto_arima` (search automatica guidata);
- `statsmodels.SARIMAX` (controllo più esplicito e diagnostica completa).

---

## 11) Diagnostica residuale
Vengono richiamati i grafici classici:
- residui nel tempo (media circa zero, varianza stabile);
- densità residui (vicina alla normale);
- Q-Q plot (allineamento circa lineare);
- ACF residui (assenza di autocorrelazioni significative).

Significato didattico:
- se il residuo “ha ancora struttura”, il modello non ha spiegato tutta la dinamica;
- forecast numericamente buono ma con residui strutturati è spesso fragile fuori campione.

---

## 12) SARIMAX: variabili esogene
SARIMAX aggiunge regressori esterni `X` sincronizzati temporalmente con la serie target.

Quando è fondamentale:
- sai che driver esterni influenzano la serie (es. promozioni, meteo, prezzi, calendario).

Chiarimento importante:
- la serie principale è “endogenous”;
- gli esterni sono “exogenous” e non vengono modellati come AR/MA interni, ma come input espliciti.

---

## 13) Intervalli di previsione
Le slide ricordano che forecast puntuale non basta.

Da prospettiva ingegneristica:
- devi comunicare **incertezza** oltre al valore centrale.
- in pratica si usa una distribuzione ipotizzata (spesso normale) e intervalli tipo `\mu \pm \sigma`.

Viene proposto anche un approccio operativo quando API non fornisce CI affidabili:
- stimare varianze per orizzonte (lag) via finestre scorrevoli storiche;
- costruire intervalli per ogni passo di forecast.

---

## 14) SES (Simple Exponential Smoothing)
Equazione base:
\[
\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t
\]

Interpretazione:
- pesa di più osservazioni recenti se `\alpha` alto;
- pesa più storia se `\alpha` basso.

Limite:
- non gestisce bene trend/stagionalità marcati.

Valore pratico:
- baseline robusta e veloce;
- spesso sorprendentemente competitiva su serie “semplici”.

---

## 15) Holt-Winters (HWES)
Estende SES con livello + trend + stagionalità (additiva o moltiplicativa).

Punti chiave:
- parametri `\alpha`, `\beta`, `\gamma`;
- adatto a serie con trend e/o stagionalità;
- implementazione disponibile in `statsmodels.ExponentialSmoothing`.

Messaggio metodologico:
- è un riferimento forte nei benchmark classici;
- spesso più stabile di modelli molto complessi su dataset piccoli/medi.

---

## 16) Theta (cenno)
La lezione cita Theta come metodo semplice ma potente (M-competitions):
- combina componenti trend/smoothed;
- può dare forecast molto competitivi con complessità limitata.

Per uno studente magistrale è un buon promemoria: **semplice non significa debole**.

---

## 17) Workflow consigliato (riassunto operativo)
1. Analizza serie: trend, stagionalità, outlier, trasformazioni log/diff.
2. Costruisci baseline semplici (SES/HWES, ARIMA base).
3. Se serve stagionalità: SARIMA.
4. Se hai driver esterni affidabili: SARIMAX.
5. Scegli ordini con AIC/AICc + validazione temporale.
6. Controlla residui (normalità, autocorrelazione, eteroschedasticità).
7. Riporta metriche (`MAE`, `RMSE`, `MAPE`) e intervalli di previsione.

---

## 18) Errori comuni evidenziati implicitamente dalle slide
- applicare AR/MA/ARMA su serie non stazionaria senza differenziare;
- scegliere modello solo per fit in-sample;
- ignorare diagnostica residui;
- usare auto-search senza controllo umano sul risultato;
- comunicare solo forecast puntuale, senza incertezza.

---

## 19) Conclusione didattica
Questa lezione costruisce una “spina dorsale” del forecasting classico:
- parti da modelli lineari interpretabili;
- introduci differenziazione e stagionalità in modo rigoroso;
- usi criteri informativi e diagnostica per validare;
- estendi con esogene quando il dominio lo richiede.

In una tesi/progetto magistrale, questa impostazione è ottima perché è:
- formalmente solida;
- implementabile in modo riproducibile;
- difendibile in discussione tecnica.
