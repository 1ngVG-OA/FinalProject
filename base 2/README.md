# Mobile Screen Time Forecasting (2020–2023)

This project leverages statistical models and neural networks to analyze and forecast daily smartphone screen time usage from 2020 to 2023. The dataset is cleaned, explored, and modeled using various techniques to assess and compare forecasting performance.

## Project Goals

* Time series analysis of smartphone screen time
* Data cleaning (handling outliers and missing values)
* Trend and seasonality decomposition
* Short- and medium-term forecasting using:

  * Statistical models: SARIMA, Holt-Winters
  * Neural networks: LSTM, MLP
  * Regressors: XGBoost, Random Forest

## Technologies Used

* Python (Pandas, NumPy, Matplotlib, Scikit-learn)
* Statsmodels, pmdarima
* PyKalman for Kalman filtering
* Torch (PyTorch) for LSTM/MLP neural models
* XGBoost and Random Forest

## 📊 Results

Various forecasting models were evaluated using RMSE on an independent test set. The results are:

| Model                       | Frequency  | RMSE      |
| --------------------------- | ---------- | --------- |
| SARIMA                      | Monthly    | 21.01     |
| Holt-Winters                | Monthly    | 25.51     |
| LSTM                        | Monthly    | 15.92     |
| MLP                         | Monthly    | 50.53     |
| XGBoost                     | Monthly    | 35.40     |
| Random Forest Regressor     | Monthly    | 35.62     |
| SARIMA                      | Weekly     | 28.47     |
| Holt-Winters                | Weekly     | 28.33     |
| LSTM                        | Weekly     | 28.65     |
| MLP                         | Weekly     | 36.39     |
| XGBoost                     | Weekly     | 25.03     |
| **Random Forest Regressor** | **Weekly** | **15.37** |
| Holt-Winters                | Original   | 87.99     |
| MLP                         | Original   | 87.97     |
| Random Forest Regressor     | Original   | 24.27     |

## 📈 Outputs

The script generates a wide range of visualizations, including:

* Daily, weekly, and monthly screen time trends
* Model diagnostics
* Forecasts compared to actual usage data
* Models comparations

## Implemented Models

* **Statistical**: ARIMA/SARIMA, Holt-Winters
* **Neural**: LSTM, MLP
* **Tree-based Regressors**: XGBoost, Random Forest
* Supports **monthly and weekly resampling**
* Advanced cleaning using **Kalman filtering** and time-based interpolation
