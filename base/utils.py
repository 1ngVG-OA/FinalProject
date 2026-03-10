"""
Utility comuni per analisi time series, visualizzazione e valutazione.

Collegamento slide:
- Slide 3: ACF/PACF per diagnostica dei modelli statistici.
- Slide 6-7: metriche descrittive/inferenziali per confronto forecast.
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def plot_acf_pacf(series, title='', layout='horizontal', figsize=(12, 4)):
    """
    Mostra ACF e PACF della serie.

    Uso pratico:
    - ACF: persistenza/autocorrelazioni a vari lag.
    - PACF: correlazioni "pulite" condizionate ai lag intermedi.

    Questi grafici aiutano la scelta preliminare di ordini AR/MA e
    identificano stagionalità (picchi a lag multipli del periodo).
    """
    if layout == 'horizontal':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    elif layout == 'vertical':
        fig, axes = plt.subplots(2, 1, figsize=figsize)
    else:
        raise ValueError("wrong layout value")

    plot_acf(series, ax=axes[0])
    axes[0].set_title('Autocorrelation (ACF)')

    plot_pacf(series, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation (PACF)')

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    plt.show()


def plot_predictions(ts, val_predictions, test_predictions, title=None):
    """
    Confronta visivamente serie originale e previsioni su validation/test.

    Utile per:
    - verificare bias sistematici,
    - individuare ritardi di fase,
    - controllare under/over-shoot su picchi e cambi di regime.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ts, label='Original')
    plt.plot(val_predictions, label='Validation')
    plt.plot(test_predictions, label='Test')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def evaluate_metrics(true_values, predicted_values, verbose=True):
    """
    Calcola MAE, RMSE e MAPE.

    - MAE: errore medio assoluto (scala originale).
    - RMSE: enfatizza errori grandi (quadratico).
    - MAPE: errore percentuale medio (utile per confronti relativi).

    Restituisce una tupla (mae, rmse, mape).
    """
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = root_mean_squared_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
    
    if verbose:
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    return mae, rmse, mape
