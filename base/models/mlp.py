"""
Modulo MLP per forecasting univariato.

Ruolo nel progetto:
- implementa la famiglia "Predictive neural models".

Collegamento slide:
- Slide 4: reti neurali feed-forward per pattern non lineari.
- Slide 2: necessità di scaling dei dati in input.
- Slide 8: ricerca iperparametri (look_back, hidden_size, lr, ecc.).

Tecnologie:
- PyTorch (definizione modello + training),
- scikit-learn (StandardScaler, ParameterGrid),
- NumPy/Pandas per manipolazione dati.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

def seed_everything(seed):
    """Imposta seed per risultati riproducibili nel training neurale."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def scale_series(train, val):
    """
    Standardizza train e validation usando la media/deviazione del train.

    Perché:
    - le MLP convergono meglio con feature scalate;
    - evita leakage perché il fit dello scaler avviene solo sul train.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).ravel()
    val_scaled = scaler.transform(val.reshape(-1, 1)).ravel()
    return train_scaled, val_scaled, scaler


def inverse_transform(scaler, data):
    """Riporta le predizioni scalate all'unità originale della serie."""
    return scaler.inverse_transform(data.reshape(-1, 1)).ravel()


def plot_series(true, predicted, title):
    """Utility di plotting locale (debug/analisi rapida)."""
    plt.figure(figsize=(10, 5))
    plt.plot(true, "o-", label="True")
    plt.plot(predicted, "x-", label="Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def create_dataset(arrdata, look_back=1):
    """
    Trasforma la serie in dataset supervisionato (X, y) con finestra mobile.

    Esempio:
    - look_back=3
    - X[i] = [x_t, x_{t+1}, x_{t+2}]
    - y[i] = x_{t+3}

    Concetto chiave delle slide neurali: convertire il forecasting in regressione.
    """
    dataX, dataY = [], []
    for i in range(len(arrdata) - look_back):
        a = arrdata[i:(i + look_back)]
        dataX.append(a)
        dataY.append(arrdata[i + look_back])
    return np.array(dataX), np.array(dataY)


def prepare_tensors(series, look_back):
    """Converte il dataset supervisionato in tensori PyTorch."""
    x, y = create_dataset(series, look_back)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


class MLP(nn.Module):
    """
    MLP minimale per regressione time series one-step.

    Architettura:
    input_size -> hidden_size -> output(1)

    - attivazione configurabile (ReLU/Tanh)
    - dropout opzionale per regolarizzazione
    """

    def __init__(self, input_size, hidden_size, activation="relu", dropout=0.0):
        super().__init__()
        act_fn = {"relu": nn.ReLU(), "tanh": nn.Tanh()}.get(activation, nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


from torch.utils.data import DataLoader, TensorDataset

def train_model(
    model, x_train, y_train, epochs, lr, batch_size=32, patience=20, verbose=False
):
    """
    Addestra il modello MLP con Adam + MSE e early stopping sul loss di training.

    Nota tecnica:
    - l'early stopping qui controlla solo la loss su train;
    - in setup più avanzati si monitorerebbe una validation loss separata.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_loss = epoch_loss / len(loader.dataset)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    return best_loss



def recursive_forecast(model, initial_window, steps):
    """
    Forecast multi-step ricorsivo.

    Workflow:
    1) predice il prossimo valore dalla finestra corrente;
    2) inserisce la predizione in coda alla finestra;
    3) ripete per il numero di step richiesto.

    Questo schema è comune in forecasting quando il modello è one-step.
    """
    model.eval()
    preds = []
    window = initial_window.clone()
    for _ in range(steps):
        with torch.no_grad():
            pred = model(window.unsqueeze(0)).item()
        preds.append(pred)
        window = torch.cat([window[1:], torch.tensor([pred])])
    return np.array(preds)


def train_and_evaluate_grid(train_series, val_series, param_grid, epochs, seed=None, verbose=False):
    """
    Esegue grid search su iperparametri MLP e valuta RMSE su validation.

    Restituisce DataFrame ordinato per RMSE crescente.
    Corrisponde al concetto di parameter fitting delle slide.
    """
    if seed != None:
        seed_everything(seed)
    results = []
    for params in ParameterGrid(param_grid):
        look_back = params["look_back"]
        if verbose:
            print(f"Testing config: {params}")

        train_scaled, val_scaled, scaler = scale_series(train_series, val_series)
        x_train, y_train = prepare_tensors(train_scaled, look_back)
        initial_window = torch.tensor(train_scaled[-look_back:], dtype=torch.float32)

        model = MLP(
            look_back,
            params["hidden_size"],
            params["activation"],
            params["dropout"],
        )

        train_model(
            model,
            x_train,
            y_train,
            epochs,
            params["lr"],
            batch_size=params.get("batch_size", 32),  # default 32
            patience=10,
        )

        preds_scaled = recursive_forecast(model, initial_window, len(val_scaled))
        preds = inverse_transform(scaler, preds_scaled)
        rmse_val = sqrt(mse(val_series, preds))

        results.append({**params, "rmse": rmse_val, "preds": preds})

    return pd.DataFrame(results).sort_values("rmse")



def train_final_and_evaluate(train_val, test, best_params, epochs, seed=None):
    """
    Riaddestra il modello sui dati train+validation con i best_params
    e genera forecast sul test set.

    È la fase finale standard dopo il tuning su validation.
    """
    if seed != None:
        seed_everything(seed)
    train_scaled, test_scaled, scaler = scale_series(train_val.values, test.values)
    x_train, y_train = prepare_tensors(train_scaled, best_params["look_back"])

    model = MLP(
        best_params["look_back"],
        best_params["hidden_size"],
        best_params["activation"],
        best_params["dropout"],
    )

    train_model(
        model,
        x_train,
        y_train,
        epochs,
        float(best_params["lr"]),
        patience=10,
    )

    initial_window = torch.tensor(train_scaled[-best_params["look_back"] :], dtype=torch.float32)
    preds_scaled = recursive_forecast(model, initial_window, len(test_scaled))
    preds = inverse_transform(scaler, preds_scaled)
    return preds
