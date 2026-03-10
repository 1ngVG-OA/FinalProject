# Ogrin Time Series Forecasting Results Report

## Overview
This report presents the results of a comprehensive time series forecasting analysis on a financial time series regarding the historical price trends of Ogrins, a virtual currency used in the MMORPG Dofus, comparing four different modeling approaches: ARMA, Multi-Layer Perceptron (MLP), and XGBoost.

## Dataset Information
- **Source File:** ogrin.csv
- **Training samples:** 106
- **Validation samples:** 20
- **Test samples:** 16
- **Total observations:** 152

## Data Preprocessing Pipeline
- **Scaling:** Applied only for MLP model (Standard scaler)

## Model Configurations

### 1. ARMA (AutoRegressive Moving Average)
**Model Type:** Statistical time series model

> *The hyperparameters of the model were selected automatically using the auto_arima function from the pmdarima library with the differencing parameter manually set to zero (d=0). The Augmented Dickey-Fuller (ADF) test confirmed that the series is stationary (p-value < 0.05), and autocorrelation plots indicated no significant seasonality. Therefore, the data was modeled without seasonal components or differencing, resulting in an ARMA(p,q) specification. *

**Final selected model:**

- **ARMA Order:** (1, 0)
- **Parameter Selection:** Auto-optimized using AIC criterion

**Performance Metrics:**
| Metric    | Validation | Test    |
|-----------|------------|---------|
| MAE       | 39.2794    | 74.1900 |
| RMSE      | 46.7457    | 91.3490 |
| MAPE (%)  | 7.22       | 16.34   |


### 2. MLP (Multi-Layer Perceptron)
**Model Type:** Neural network regression model

> *The architecture and hyperparameters of the model were selected through a grid search, exploring different configurations of look-back window, hidden layer size, dropout rate, batch size, learning rate, and activation function.*

#### Grid Search Parameter Space

- **Look-back window:** {7, 14, 21, 28, 30}  
- **Hidden layer size:** {2, 4, 6, 8}  
- **Learning rate:** {0.01, 0.001, 0.0001}  
- **Dropout:** {0.0, 0.1}  
- **Batch size:** {8, 16, 32}  
- **Activation function:** {ReLU, tanh}  


> *The final configuration was selected based on the lowest RMSE obtained on the validation set.*

**Final selected model:**

- **Architecture:** Input Layer → Dense (6 units, tanh) → Dense (1 unit, Linear)  
- **Look-back window:** 28 days  
- **Training epochs:** 500  
- **Batch size:** 32
- **Optimizer:** Adam with learning rate = 0.001  
- **Activation function:** tanh 
- **Loss function:** Mean Squared Error (MSE)

**Performance Metrics:**  
| Metric   | Validation | Test     |
|----------|------------|----------|
| MAE      | 44.6773    | 96.2034  |
| RMSE     | 56.3985    | 116.6694 |
| MAPE (%) | 7.73       | 20.84    |

### 3. XGBoost (Extreme Gradient Boosting)
**Model Type:** Ensemble tree-based regression model

> *The model's hyperparameters were selected through a grid search exploring various configurations of look-back window, number of trees, tree depth, learning rate (eta), and subsampling ratios.*

#### Grid Search Parameter Space

- **Look-back window:** {7, 14, 21, 28, 30}  
- **Number of estimators (trees):** {5, 10, 15, 20, 30, 50, 80, 100}  
- **Max depth:** {1, 3, 5, 7, 9}
- **Learning rate (eta):** {0.1, 0.2, 0.4, 0.6, 0.8, 1}  
- **Subsample:** {0.1, 0.3, 0.5, 0.7, 0.9, 1}  
- **Colsample bytree:** {0.1, 0.3, 0.5, 0.7, 0.9, 1}  
- **Random seed:** 1234  

> *The best-performing hyperparameter combination was chosen based on the lowest RMSE on the validation set.*

**Final selected model:**

- **Look-back window:** 28 days  
- **Number of estimators (trees):** 80
- **Max depth:** 5
- **Learning rate (eta):** 1
- **Subsample:** 0.7  
- **Colsample bytree:** 1

**Performance Metrics:**  
| Metric   | Validation | Test     |
|----------|------------|----------|
| MAE      | 28.8962    | 64.0156  |
| RMSE     | 35.1934    | 78.7274  |
| MAPE (%) | 5.17       | 13.86    |


## Comparative Analysis

### Validation Set Performance
| Model    | RMSE     | MAE      | MAPE (%) | Rank by RMSE |
|----------|----------|----------|----------|--------------|
| XGBoost  | 35.1934  | 28.8962  | 5.17     | 1            |
| ARMA     | 46.7457  | 39.2794  | 7.22     | 2            |
| MLP      | 56.3985  | 44.6773  | 7.73     | 3            |

### Test Set Performance (Final Evaluation)
| Model    | RMSE      | MAE      | MAPE (%) | Rank by RMSE |
|----------|-----------|----------|----------|--------------|
| XGBoost  | 78.7274   | 64.0156  | 13.86    | 1            |
| ARMA     | 91.3490   | 74.1900  | 16.34    | 2            |
| MLP      | 116.6694  | 96.2034  | 20.84    | 3            |

## Key Results

### Best Performing Model
**XGBoost** emerged as the top-performing model, achieving the lowest error rates across all evaluation metrics.
